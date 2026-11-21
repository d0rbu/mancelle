"""General-validation BPB under steering+adversarial interventions (opt-in).

This is the ``score_general_val`` stage. It answers a question the concept-margin
metric cannot: how much does a steering signal (and the adversarial pressure used
to break it) degrade the model's GENERAL language-modelling ability? The concept
margin only sees concept-specific completions; here we score a held-out GENERIC
validation set (FineWeb-Edu held-out shard, the pretraining distribution) under
the SAME intervention used for the margin.

Design — reuse, don't reinvent:

* The scored quantity is the model's teacher-forced BPB on generic text under an
  intervention. We reuse the EXACT scoring kernel the concept margin uses
  (``steering.eval.axbench_bpb.teacher_force.score_triple``); the only difference
  is the input text (generic val docs instead of concept completions). So
  ``general_val_bpb`` is an apples-to-apples companion to ``primary_margin_y``.

* The steered intervention depends only on ``(target_model, method_name,
  concept_id, coefficient_str, adversarial_method, adversarial_strength)`` — NOT
  on the prefix or the completion. So we build interventions ourselves with a
  deterministic seed (independent of any ``core_prefix_id``) and score the generic
  val set once per unique intervention SIGNATURE.

* The UNIVERSAL Faster-GCG attack makes general_val ATTACK-AWARE. A concept's
  optimized k-token trigger (read from ``universal_triggers.parquet``) is
  substituted into the last-k tokens of every DECODER-ONLY validation doc, so
  ``general_val_bpb`` measures the general LM ability UNDER the same attack. The
  trigger differs per ``strength_index/k`` (the objective is
  ``grad_contrast - grad_desired``), so ``general_val_bpb`` is now NON-FLAT
  across the GCG axis. The signature forks on ``faster_gcg:k{strength_index}``
  for that reason; k=0 (no trigger) is the clean baseline.
  GEPA carries no universal trigger and stays on the shared ``base`` intervention;
  only the additive counter-direction (a residual hook, magnitude > 0) moves the
  steered intervention itself. The ATTACKED docs are DECODER-ONLY (encdec
  general_val attack is deferred); encdec still scores the clean docs.

The result table (``GENERAL_VAL_BPB_SCHEMA``) carries one row per distinct
intervention combination present in the candidates; aggregation left-joins it onto
the macro rows (keyed on the intervention identity) and
concept-averages it. See ``steering/eval/axbench_bpb_adv/aggregation.py``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet
from nanochat.dataset import DatasetSplit, parquets_iter_batched
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.percentile_table import build_percentile_table
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.method_registry import BaselineKind, get_method_by_name
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.context_cache import extract_prefix_features
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle, score_triple
from steering.eval.axbench_bpb.tokenization import truncate_prefix_left
from steering.eval.axbench_bpb.types import (
    Intervention,
    LoraParams,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.axbench_bpb.vocab_restriction import (
    UNCOMPACTED_VOCAB_FINGERPRINT,
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    assert_table_cluster_identity,
    bind_clustered_table,
    load_clustered_candidate_snapshot,
    write_clustered_table_atomic,
)
from steering.eval.axbench_bpb_adv.bos_utils import (
    assert_bos_padded_tokens,
    assert_no_bos_payload,
)
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    extract_raw_clustered_prefix,
    reconstruct_clustered_replay_state,
    select_raw_clustered_segment,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    AxBenchBpbAdvConfig,
    OursGcgConfig,
    get_additive_direction_config,
)
from steering.eval.axbench_bpb_adv.editable_tokens import (
    replace_editable_tokens,
    select_final_k_editable_span,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    read_evaluation_population,
    require_table_in_evaluation_population,
)
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    GeneralValSignalBudget,
    GeneralValWarmStartCaps,
    build_general_val_doc_member,
    derive_real_output_signal,
    general_val_bag_identity_metadata,
    general_val_doc_gcg_config,
    optimize_general_val_doc_bag,
    penalized_lm_objective_value,
    validate_general_val_bag_metadata,
)
from steering.eval.axbench_bpb_adv.intervention_build import (
    apply_additive_counter,
    attach_additive_counter,
    resolve_additive_counter,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    VOCAB_FINGERPRINT_META_KEY,
    read_ours_gcg_bag,
    write_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    ROOFLINE_PRECOMPUTED_BAG_FAMILIES,
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    GENERAL_VAL_BPB_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    _bpb_config_from_adv_config,
    _build_token_bytes,
    _concepts_from_entries,
    _preflight_direction_scales_for_rows,
    _read_concepts,
    load_candidate_table,
)
from steering.eval.axbench_bpb_adv.universal_gcg import ConceptGroupKey, group_key
from steering.eval.segment_mode_guard import require_fixed_feature_dir_segment_mode
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.sae_resolver import parse_neuronpedia_source
from steering.segmentation import resolve_bos_token_id
from steering.stats_io import load_statistics
from steering.stats_types import ExtractionStatistics

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.source_validation import (
        BpbAdvSourceValidation,
    )
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

# Placeholder labels for the concept-independent generic-val samples. The BPB
# kernel copies concept_id/mode/content_class onto its score row but never uses
# them for the loss math, so a fixed placeholder is safe (and clearer than a
# misleading per-intervention concept_id, since the SAME samples are scored under
# every concept's intervention).
_GENERAL_VAL_CONCEPT_ID = -1
_GENERAL_VAL_CONTENT_CLASS = "general_val"


@dataclass(frozen=True)
class _GeneralValDoc:
    """One generic validation document, sliced into a prefix + scored output.

    Token ids are the nanochat (decoder) tokenizer's. For clean docs,
    ``prefix_text`` is the BOS-free decoded content prefix used to extract Gemma
    SAE features for the encdec encoder; decoder-only attacked docs may re-decode
    the attacked BOS-prefixed ids because they do not use context segments.
    ``output_byte_weights`` is the per-output-token UTF-8 byte length (for BPB
    normalization).
    """

    sample_id: str
    prefix_tokens: list[int]
    output_tokens: list[int]
    prefix_text: str
    output_byte_weights: t.Tensor


@dataclass(frozen=True)
class _InterventionCombo:
    """One distinct intervention coordinate present in the candidates.

    ``strength_index`` (the k-grid ordinal) is part of the coordinate: under the
    universal Faster-GCG trigger attack the substituted trigger -- hence
    ``general_val_bpb`` and the dedup SIGNATURE -- differs per ``k``. Two combos
    that differ ONLY in ``strength_index`` must not collapse (else a k silently
    loses its row).
    """

    target_model: str
    method_name: str
    mode: str
    concept_id: int
    coefficient_str: str
    adversarial_method: str
    adversarial_strength: float
    strength_index: int


def iter_general_val_texts() -> Iterator[str]:
    """Yield held-out FineWeb-Edu documents (val split) in deterministic order."""
    for batch in parquets_iter_batched(DatasetSplit.VAL):
        for text in batch:
            if text and text.strip():
                yield str(text)


def build_general_val_docs(
    *,
    texts: Iterable[str],
    tokenizer: Any,
    token_bytes: t.Tensor,
    n_documents: int,
    prefix_tokens: int,
    output_tokens: int,
) -> list[_GeneralValDoc]:
    """First ``n_documents`` docs long enough to slice (D5: n = SCORED docs).

    The prefix is BOS-ADDITIVE: ``prefix = [BOS] + real[:prefix_tokens]`` and
    ``output = real[prefix_tokens : prefix_tokens + output_tokens]`` where
    ``real`` is the raw (no-BOS) encoding. The BOS is purely additive -- it adds
    a leading token WITHOUT consuming a real-token slot, so the scored OUTPUT
    continuation is the SAME real tokens it would have been without BOS (the M4
    contract: prepending BOS must never shift the scored window). This aligns the
    validation docs with the BOS-prefixed context in which the steering
    directions and the universal trigger were computed, so the trigger transfer
    is faithful. NOTE: this intentionally shifts the CLEAN general_val baseline
    relative to the old no-BOS docs -- it is the more-correct behavior.

    ``texts`` is streamed (not materialized): documents are pulled one at a time
    and scanned until ``n_documents`` have survived slicing (or the stream is
    exhausted). This makes doc SELECTION independent of how many candidate texts
    happen to be available -- the old contract fetched exactly ``n_documents``
    candidates up front and silently scored fewer when some were too short; this
    one always scores exactly ``n_documents`` docs, or raises.

    Documents whose RAW (no-BOS) encoding is too short to supply
    ``prefix_tokens + output_tokens`` tokens are skipped. The prefix is decoded
    back to text for Gemma feature extraction.
    """

    # Shared BOS-resolution (probes both the .bos_token_id attribute and
    # get_bos_token_id()); the candidate stage resolves the same way via
    # optimize._bos_token_id, so the two BOS-prefixed contexts agree.
    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise AxBenchBpbAdvError(
            "general_val_bpb: tokenizer has no discoverable BOS token id; cannot "
            "BOS-prefix the validation docs (the docs must match the BOS-prefixed "
            "direction/trigger context)"
        )
    docs: list[_GeneralValDoc] = []
    scanned = 0
    for i, text in enumerate(texts):
        scanned += 1
        # Raw (no-BOS) encoding: BOS is added additively below so the scored
        # OUTPUT window is NOT shifted by the prepended BOS (M4).
        real = [int(tid) for tid in tokenizer.encode(text)]
        if real and real[0] == bos_id:
            raise AxBenchBpbAdvError(
                "general_val_bpb: raw tokenizer encoding for validation document "
                f"{i} already starts with BOS token id {bos_id}; refusing to "
                "prepend another BOS and create a double-BOS prefix"
            )
        if len(real) < prefix_tokens + output_tokens:
            continue
        # Prepend BOS, mirroring encode_prefix: the nanochat decoder (and the
        # concept-margin scorer + base_eval's dataloader) ALWAYS score with a
        # leading BOS. Without it the model runs off-distribution and the output
        # BPB blows up far past the uniform-random ceiling (ln(vocab)). prefix_text
        # stays BOS-free — it feeds the Gemma extraction tokenizer for the encdec
        # context, which prepends its own BOS.
        content_prefix_ids = real[:prefix_tokens]
        prefix_ids = assert_bos_padded_tokens(
            [bos_id, *content_prefix_ids],
            bos_id=bos_id,
            context=f"general_val_bpb document {i} prefix",
        )
        output_ids = real[prefix_tokens : prefix_tokens + output_tokens]
        output_tensor = t.tensor(output_ids, dtype=t.long)
        try:
            byte_weights = token_bytes[output_tensor]
        except IndexError:
            continue
        if int(byte_weights.sum().item()) <= 0:
            continue
        docs.append(
            _GeneralValDoc(
                sample_id=f"general_val_{i}",
                prefix_tokens=prefix_ids,
                output_tokens=output_ids,
                prefix_text=str(tokenizer.decode(content_prefix_ids)),
                output_byte_weights=byte_weights,
            )
        )
        if len(docs) >= n_documents:
            break
    if len(docs) < n_documents:
        if scanned == 0:
            raise AxBenchBpbAdvError(
                "general_val_bpb is enabled but no FineWeb-Edu validation documents "
                "were found in base_data. Download a shard with "
                "`python -m nanochat.dataset -n 1` (the last shard is the val split)."
            )
        raise AxBenchBpbAdvError(
            f"general_val_bpb: found {len(docs)} of {n_documents} requested "
            f"validation documents long enough for prefix_tokens={prefix_tokens} "
            f"+ output_tokens={output_tokens} after scanning {scanned} texts; "
            "lower those knobs or provide more/longer documents"
        )
    logger.info(f"general_val_bpb: scanned {scanned} texts to select {len(docs)} scored docs")
    record_counter("axbench_bpb_adv.general_val.docs_scanned", scanned)
    record_counter("axbench_bpb_adv.general_val.docs_selected", len(docs))
    return docs


def _doc_to_scoring_input(
    doc: _GeneralValDoc,
    *,
    segments: tuple[ContextFeatureSegment, ...],
    max_seq_len: int,
) -> ScoringInputs:
    _, truncated = truncate_prefix_left(
        prefix=doc.prefix_tokens,
        n_output=len(doc.output_tokens),
        max_seq_len=max_seq_len,
    )
    return ScoringInputs(
        sample_id=doc.sample_id,
        concept_id=_GENERAL_VAL_CONCEPT_ID,
        mode=SteeringMode.POSITIVE,
        content_class=_GENERAL_VAL_CONTENT_CLASS,
        prefix_text=doc.prefix_text,
        prefix_hash=make_prefix_hash(doc.prefix_text),
        prefix_tokens=doc.prefix_tokens,
        output_tokens=doc.output_tokens,
        prefix_context_segments=segments,
        n_prefix=len(doc.prefix_tokens),
        n_output=len(doc.output_tokens),
        output_byte_weights=doc.output_byte_weights,
        truncated=truncated,
    )


@dataclass
class _RooflineGeneralValState:
    """All mutable state required by roofline general-val scoring."""

    signal_budget: GeneralValSignalBudget
    real_bags: dict[str, FeatureConditioning]
    bag_config_metadata: dict[str, str]

    @classmethod
    def for_budget(cls, signal_budget: GeneralValSignalBudget) -> _RooflineGeneralValState:
        return cls(
            signal_budget=signal_budget,
            real_bags={},
            bag_config_metadata={},
        )


def _extract_with_target_memory_handoff(
    *,
    runtime_state: RuntimeState,
    target_bundles: dict[str, TargetModelBundle],
    extract: Callable[[], dict[str, tuple[ContextFeatureSegment, ...]]],
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    """Run Gemma+SAE extraction without co-resident target checkpoints.

    The Gemma-3-4B extractor, 65k SAE, decoder, and encoder-decoder target do
    not fit together on a 24 GiB device.  Drop every target owner before
    extraction, then release *both* extraction owners and reload the exact
    pinned targets before scoring.  Cleanup runs even when extraction fails;
    target restoration does not, so the original exception cannot be masked by
    an unnecessary reload.
    """

    target_names = tuple(target_bundles)
    unsupported = set(target_names) - {"decoder_only", "encdec"}
    if unsupported:
        raise AxBenchBpbAdvError(
            "general_val_bpb: cannot restore unsupported target bundle(s) "
            f"{sorted(unsupported)!r} after extraction"
        )

    # This must precede RuntimeState.release_target_models(): the caller-owned
    # bundles contain the same modules and their clustered pre-forward guards
    # close over those modules.  Retaining either owner defeats the handoff.
    target_bundles.clear()
    runtime_state.release_target_models()
    release_cuda_iteration_memory()
    try:
        result = extract()
    except BaseException:
        # A bare `finally:` here would let a cleanup failure REPLACE the
        # extraction error -- the exact masking this function's contract says it
        # avoids. Cleanup still runs on the failure path, but its own exception
        # is logged rather than raised so the original propagates. On the success
        # path a cleanup failure is still fatal (unchanged).
        try:
            runtime_state.release_base_gemma_and_sae()
            release_cuda_iteration_memory()
        except Exception:
            logger.exception(
                "general_val_bpb: extraction cleanup failed while an extraction "
                "error was already in flight; preserving the original error"
            )
        raise
    runtime_state.release_base_gemma_and_sae()
    release_cuda_iteration_memory()
    for target in target_names:
        if target == "decoder_only":
            target_bundles[target] = runtime_state.decoder_only()
        else:
            target_bundles[target] = runtime_state.encdec()
    return result


def _extract_encdec_segments(
    docs: list[_GeneralValDoc],
    *,
    runtime_state: RuntimeState,
    cfg: AxBenchBpbAdvConfig,
    statistics: ExtractionStatistics,
    roofline_state: _RooflineGeneralValState | None = None,
    tokenizer: Any | None = None,
    device: t.device | None = None,
    allowed_ids: t.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    corpus_signature: str | None = None,
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    """Extract per-doc Gemma SAE feature segments for the encdec encoder.

    Mirrors the AxBench-BPB context-cache stage (sae_layer + boundaries +
    ``extract_prefix_features``) but for generic val prefixes. Offloads the base
    Gemma model to CPU afterwards so the target models have room to score.

    Every scored doc must condition: a doc whose extraction returns an empty
    segment tuple raises immediately (D7) rather than being silently skipped,
    which used to fork the encdec corpus from the decoder corpus and break
    frontier comparability (issue #635).

    When ``roofline_state`` is provided, this also derives each doc's real
    output-window conditioning signal before offloading Gemma. The state's real
    bag dict is mutated in place and keyed by ``doc.sample_id``.

    ``allowed_ids`` restricts that real-signal derivation to a compacted target
    checkpoint's allowed SAE feature vocabulary ``V`` (see
    :func:`steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`);
    it is threaded straight through to :func:`derive_real_output_signal`.
    ``None`` (the default, an uncompacted checkpoint) reproduces the
    unrestricted selection byte-for-byte.
    """

    if roofline_state is not None and (tokenizer is None or device is None):
        raise AxBenchBpbAdvError(
            "general_val_bpb: roofline real-signal extraction requires both tokenizer and device"
        )
    _release, _act, sae_layer, _width, _tier = parse_neuronpedia_source(cfg.neuronpedia_source)
    if cluster_contract is None:
        base_model, base_sae = runtime_state.base_gemma_and_sae()
    else:
        base_model, base_sae = runtime_state.base_gemma_and_sae(
            offload_transformer_tail_after=sae_layer,
        )
    boundaries = build_boundaries_from_stats(
        statistics.per_feature_stats,
        n_sae_features=base_sae.cfg.d_sae,
        num_buckets=cfg.context_cache.num_buckets,
    )
    percentile_table: t.Tensor | None = None
    clustered_replay = None
    if cluster_contract is not None:
        if corpus_signature is None:
            raise AxBenchBpbAdvError(
                "clustered general_val context extraction requires the exact "
                "generic corpus signature"
            )
        if int(cfg.context_cache.segment_size) != int(
            cluster_contract.feature_directory.segment_size
        ):
            raise AxBenchBpbAdvError(
                "clustered general_val context segment size does not match the "
                "validated feature contract"
            )
        if int(cfg.context_cache.num_buckets) != int(cluster_contract.selection.num_buckets):
            raise AxBenchBpbAdvError(
                "clustered general_val context bucket count does not match the "
                "validated checkpoint selection"
            )
        if float(cfg.extraction_threshold) != 0.0:
            raise AxBenchBpbAdvError(
                "clustered general_val raw extraction requires extraction_threshold=0.0"
            )
        clustered_replay = reconstruct_clustered_replay_state(cluster_contract)
    if roofline_state is not None:
        percentile_table = build_percentile_table(
            statistics.per_feature_stats,
            n_sae_features=base_sae.cfg.d_sae,
        )
        roofline_state.bag_config_metadata.clear()
        roofline_state.bag_config_metadata.update(
            _roofline_bag_config_metadata(
                cfg,
                boundaries=boundaries,
                sae_layer=sae_layer,
            )
        )
    segments_by_sample: dict[str, tuple[ContextFeatureSegment, ...]] = {}
    for doc in docs:
        if cluster_contract is None:
            segments = extract_prefix_features(
                prefix_text=doc.prefix_text,
                base_model=base_model,
                base_sae=base_sae,
                boundaries=boundaries,
                segment_size=cfg.context_cache.segment_size,
                sae_layer=sae_layer,
                extraction_max_tokens=cfg.extraction_max_tokens,
                extraction_threshold=cfg.extraction_threshold,
            )
        else:
            assert clustered_replay is not None, (
                "clustered general_val extraction requires reconstructed replay state"
            )
            raw_prefix = extract_raw_clustered_prefix(
                prefix_text=doc.prefix_text,
                base_model=base_model,
                base_sae=base_sae,
                contract=cluster_contract,
                segment_size=cfg.context_cache.segment_size,
                extraction_max_tokens=cfg.extraction_max_tokens,
            )
            clustered_segments: list[ContextFeatureSegment] = []
            for raw_segment in raw_prefix.segments:
                selected = select_raw_clustered_segment(
                    raw_segment,
                    replay_state=clustered_replay,
                    contract=cluster_contract,
                )
                feature_ids_tensor = t.cat((selected.positive_ids, selected.negative_ids)).to(
                    t.int64
                )
                negative_bucket = int(cluster_contract.selection.num_buckets) - 1
                buckets_tensor = t.cat(
                    (
                        selected.positive_buckets.to(t.int64),
                        t.full(
                            (selected.negative_ids.numel(),),
                            negative_bucket,
                            dtype=t.int64,
                        ),
                    )
                )
                cluster_contract.consolidation.assert_representatives_only(
                    feature_ids_tensor,
                    surface="clustered general_val context segment",
                )
                cluster_contract.feature_mask.assert_no_masked(feature_ids_tensor)
                clustered_segments.append(
                    ContextFeatureSegment(
                        feature_ids=tuple(int(value) for value in feature_ids_tensor.tolist()),
                        buckets=tuple(int(value) for value in buckets_tensor.tolist()),
                        char_start=raw_segment.char_start,
                        char_end=raw_segment.char_end,
                    )
                )
            segments = tuple(clustered_segments)
        # The encdec teacher-forcing kernel requires a non-empty segment tuple
        # (empty per-segment feature sets are fine; an empty tuple is not).
        # Every scored doc must condition: a per-doc skip here would fork the
        # encdec corpus from the decoder corpus and break frontier
        # comparability, so this raises immediately instead (D7) rather than
        # warning-and-dropping the doc.
        if not segments:
            raise AxBenchBpbAdvError(
                f"general_val_bpb: doc {doc.sample_id!r} produced no encdec feature "
                "segments; every scored doc must condition (raise "
                "general_val_bpb.prefix_tokens so each prefix spans >= 1 Gemma "
                "segment). Per-doc skips would fork the encdec corpus from the "
                "decoder corpus and break frontier comparability (issue #635, D7)."
            )
        segments_by_sample[doc.sample_id] = segments
        if roofline_state is not None:
            # The top-of-function guard already rejected a None device when
            # roofline_state is set; assert these so the types narrow for
            # derive_real_output_signal.
            assert tokenizer is not None
            assert device is not None
            assert percentile_table is not None
            derive_kwargs: dict[str, Any] = {
                "base_model": base_model,
                "base_sae": base_sae,
                "sae_layer": sae_layer,
                "boundaries": boundaries,
                "percentile_table": percentile_table,
                "extraction_threshold": cfg.extraction_threshold,
                "signal_budget": roofline_state.signal_budget,
                "tokenizer": tokenizer,
                "device": device,
                "allowed_ids": allowed_ids,
            }
            if cluster_contract is not None:
                derive_kwargs.update(
                    cluster_contract=cluster_contract,
                    corpus_signature=corpus_signature,
                )
            roofline_state.real_bags[doc.sample_id] = derive_real_output_signal(
                doc,
                **derive_kwargs,
            )
    # Free the base Gemma before scoring with the target models (single-GPU
    # headroom). The bundle is re-loaded lazily if a later stage needs it.
    if base_model is not None and hasattr(base_model, "to"):
        base_model.to("cpu")
        if t.cuda.is_available():
            t.cuda.empty_cache()
    if not segments_by_sample:
        # Unreachable: every doc either raises above or lands in the map, so an
        # empty map can only occur if docs itself was empty. Kept as a
        # defensive backstop (mirrors the actionable message build_general_val_
        # docs would already have raised for an empty docs list).
        raise AxBenchBpbAdvError(
            "general_val_bpb: no encdec val docs produced feature segments; "
            "raise general_val_bpb.prefix_tokens so each prefix spans >= 1 "
            "Gemma segment"
        )
    return segments_by_sample


def _build_trigger_index(
    triggers: pa.Table,
) -> dict[ConceptGroupKey, list[int]]:
    """Index the universal-trigger table by the canonical ``group_key``.

    The key is the shared 7-tuple concept-group identity (incl. ``method_name`` +
    ``target_model``): the candidate stage loops per target and per steering
    family, writing ONE trigger row per group, so the default config
    (``target_models=[decoder_only, encdec]`` × multiple decoder-only families)
    legitimately produces several rows that share the other five fields but differ
    on method/target -- those are SEPARATE groups, not a collision. Each group
    writes exactly one row, so a collision on the FULL key is a corruption of the
    artifact and raises fail-loud (a silent last-write-wins would substitute the
    wrong trigger into the validation docs)."""

    if not triggers.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA):
        raise AxBenchBpbAdvError("expected universal-trigger table with UNIVERSAL_TRIGGERS_SCHEMA")
    index: dict[ConceptGroupKey, list[int]] = {}
    for row in triggers.to_pylist():
        key = group_key(row)
        if key in index:
            raise AxBenchBpbAdvError(
                f"duplicate universal trigger for concept group {key}; the "
                "candidate stage must write exactly one trigger row per "
                "(method_name, target_model, concept, coefficient, "
                "adversarial_method, k, mode)"
            )
        index[key] = [int(token_id) for token_id in row["trigger_tokens"]]
    return index


def _lookup_universal_trigger(
    trigger_index: dict[ConceptGroupKey, list[int]],
    combo: _InterventionCombo,
) -> list[int]:
    """Resolve the unique universal trigger for a faster_gcg ``combo``.

    A missing trigger for a faster_gcg k>0 combo is a fail-loud error: every
    such combo's concept group writes a trigger row, so its absence means the
    artifact and the candidates disagree (a wrong/empty substitution would
    silently corrupt the attacked general_val)."""

    key = group_key(asdict(combo))
    trigger = trigger_index.get(key)
    if trigger is None:
        raise AxBenchBpbAdvError(
            f"no universal trigger found for faster_gcg combo {key}; the "
            f"candidate stage must write one to {UNIVERSAL_TRIGGERS_SCHEMA!r} "
            "for every faster_gcg k>0 group"
        )
    return trigger


def _attack_doc(
    doc: _GeneralValDoc,
    *,
    trigger: list[int],
    tokenizer: Any,
) -> _GeneralValDoc:
    """Substitute ``trigger`` into the BOS-prefixed doc's last-k prefix tokens.

    ``k == len(trigger)``. The trigger replaces the FINAL k tokens of the
    prefix via ``select_final_k_editable_span`` (BOS at position 0 is never
    editable -- the BOS-prefixed doc means the last-k is the tail and BOS is
    safe) + ``replace_editable_tokens`` (length-preserving). ``prefix_text`` is
    re-decoded from the attacked ids; the scored OUTPUT and its byte-weights are
    UNCHANGED. ``k == 0`` (empty trigger) is the clean baseline and returns the
    doc unchanged (no substitution)."""

    if not trigger:
        return doc
    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise AxBenchBpbAdvError(
            "general_val_bpb: tokenizer has no discoverable BOS token id; cannot "
            "validate attacked validation docs"
        )
    assert_bos_padded_tokens(
        doc.prefix_tokens,
        bos_id=bos_id,
        context=f"general_val_bpb clean doc {doc.sample_id!r}",
    )
    trigger = assert_no_bos_payload(
        trigger,
        bos_id=bos_id,
        context=f"general_val_bpb trigger for doc {doc.sample_id!r}",
    )
    span = select_final_k_editable_span(doc.prefix_tokens, k=len(trigger))
    attacked_prefix = replace_editable_tokens(doc.prefix_tokens, span, trigger)
    attacked_prefix = assert_bos_padded_tokens(
        attacked_prefix,
        bos_id=bos_id,
        context=f"general_val_bpb attacked doc {doc.sample_id!r}",
    )
    return _GeneralValDoc(
        sample_id=doc.sample_id,
        prefix_tokens=attacked_prefix,
        output_tokens=doc.output_tokens,
        prefix_text=str(tokenizer.decode(attacked_prefix)),
        output_byte_weights=doc.output_byte_weights,
    )


def _is_attacked_decoder_combo(combo: _InterventionCombo) -> bool:
    """Whether ``combo`` gets the DECODER-ONLY universal-trigger doc attack.

    Only decoder-only faster_gcg combos with a non-zero trigger length k carry a
    trigger to substitute. k is ``adversarial_strength`` (the optimized token
    count), NOT ``strength_index`` (the ``optimized_token_counts`` grid ordinal):
    the two coincide only when the grid leads with 0, so gating on the ordinal
    silently mis-scores a non-leading-0 grid (an idx-0/k>0 attack scored clean,
    or an idx>0/k=0 control sent looking for a trigger that was never written).
    k=0 is the clean control; encdec faster_gcg is deferred (clean docs);
    gepa/additive have no trigger."""

    return (
        combo.adversarial_method == AdversarialMethod.FASTER_GCG.value
        and combo.target_model == "decoder_only"
        and combo.adversarial_strength > 0.0
    )


def _samples_for_combo(
    rep: _InterventionCombo,
    *,
    docs: list[_GeneralValDoc],
    clean_samples: list[ScoringInputs],
    trigger_index: dict[tuple[Any, ...], list[int]],
    tokenizer: Any,
    decoder_max_seq_len: int,
) -> list[ScoringInputs]:
    """The scoring inputs for a signature's representative combo.

    For a decoder-only faster_gcg k>0 combo, substitute the looked-up universal
    trigger into each doc's last-k and rebuild the decoder-only ScoringInputs
    (so the scored prefix carries the attack). Otherwise return the pre-built
    CLEAN samples (k=0 control, encdec [deferred], gepa, additive, base).

    ``decoder_max_seq_len`` is the rep's target bundle max_seq_len, consumed only
    on the attacked decoder-only branch (where rep.target_model == decoder_only);
    the caller resolves it from the always-loaded bundle, so it is never absent."""

    if not _is_attacked_decoder_combo(rep):
        return clean_samples
    trigger = _lookup_universal_trigger(trigger_index, rep)
    return [
        _doc_to_scoring_input(
            _attack_doc(doc, trigger=trigger, tokenizer=tokenizer),
            segments=(),
            max_seq_len=decoder_max_seq_len,
        )
        for doc in docs
    ]


def _tensor_digest(tensor: t.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tuple(cpu.shape)).encode())
    hasher.update(str(cpu.dtype).encode())
    hasher.update(cpu.numpy().tobytes())
    return hasher.hexdigest()


def _roofline_bag_config_metadata(
    cfg: AxBenchBpbAdvConfig,
    *,
    boundaries: t.Tensor,
    sae_layer: int,
) -> dict[str, str]:
    """Identity fields that affect persisted roofline general-val bags."""

    return {
        "extraction_max_tokens": str(int(cfg.extraction_max_tokens)),
        "extraction_threshold": repr(float(cfg.extraction_threshold)),
        "neuronpedia_model_id": str(cfg.neuronpedia_model_id),
        "neuronpedia_source": str(cfg.neuronpedia_source),
        "sae_layer": str(int(sae_layer)),
        "feature_statistics_path": str(cfg.feature_statistics_path),
        "context_cache_segment_size": str(int(cfg.context_cache.segment_size)),
        "context_cache_num_buckets": str(int(cfg.context_cache.num_buckets)),
        "bucket_boundaries_shape": json.dumps(list(boundaries.shape)),
        "bucket_boundaries_dtype": str(boundaries.dtype),
        "bucket_boundaries_sha256": _tensor_digest(boundaries),
    }


def _intervention_signature(combo: _InterventionCombo) -> str:
    """Dedup key for the general_val scoring pass.

    Under the UNIVERSAL Faster-GCG trigger attack a concept's optimized trigger
    is substituted into the validation docs ONLY for the combos that actually
    carry one -- decoder-only k>0 (:func:`_is_attacked_decoder_combo`). That
    trigger differs per ``k``, so those combos FORK by
    ``faster_gcg:k{k}`` (each k scores its own
    attacked docs ⇒ a distinct ``general_val_bpb``). ``k`` here is the real
    trigger length (``adversarial_strength``), NOT the grid ordinal.

    Every NON-attacked combo scores the SAME clean BOS-prefixed docs under the
    base intervention, so they collapse to the shared ``base`` signature (scored
    once): the k=0 control, encdec faster_gcg (deferred ⇒ clean docs), GEPA (not
    a universal-GCG member ⇒ no trigger), and the additive-magnitude-0 control.
    Forking those would redundantly re-score identical clean docs per k
    and, under a finite ``max_interventions`` cap, could crowd out the real
    decoder-only attacked forks. Additive counter-steering with non-zero
    magnitude (a residual hook on the steered intervention itself) keeps its
    magnitude-keyed part.
    """

    if _is_attacked_decoder_combo(combo):
        # Per-k fork: the substituted trigger varies on k. Label by the real
        # trigger length k (adversarial_strength), not the grid ordinal.
        adv_part = f"faster_gcg:k{combo.adversarial_strength!r}"
    elif (
        combo.adversarial_method == AdversarialMethod.ADDITIVE_DIRECTION.value
        and combo.adversarial_strength > 0.0
    ):
        adv_part = f"additive:{combo.adversarial_strength!r}"
    else:
        adv_part = "base"
    family = precomputed_bag_method_family(combo.method_name)
    if family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES and not (
        combo.adversarial_method == AdversarialMethod.ADDITIVE_DIRECTION.value
        and combo.adversarial_strength > 0.0
    ):
        # Roofline general-val bags are derived from the generic validation doc,
        # not a concept, so the clean/control intervention is concept-free and can
        # be scored once then broadcast. Nonzero additive-direction rows are the
        # exception: apply_additive_counter loads a concept-specific residual
        # direction, so those signatures must keep concept_id to avoid broadcasting
        # one concept's counter hook to another.
        return (
            f"{combo.target_model}|{combo.method_name}|__roofline_doc__|"
            f"{combo.coefficient_str}|{adv_part}"
        )
    return (
        f"{combo.target_model}|{combo.method_name}|{combo.concept_id}|"
        f"{combo.coefficient_str}|{adv_part}"
    )


def _distinct_combos(candidates: pa.Table) -> list[_InterventionCombo]:
    seen: set[tuple[Any, ...]] = set()
    combos: list[_InterventionCombo] = []
    columns = [
        "target_model",
        "method_name",
        "mode",
        "concept_id",
        "coefficient_str",
        "adversarial_method",
        "adversarial_strength",
        "strength_index",
    ]
    for row in candidates.select(columns).to_pylist():
        combo = _InterventionCombo(
            target_model=str(row["target_model"]),
            method_name=str(row["method_name"]),
            mode=str(row["mode"]),
            concept_id=int(row["concept_id"]),
            coefficient_str=str(row["coefficient_str"]),
            adversarial_method=str(row["adversarial_method"]),
            adversarial_strength=float(row["adversarial_strength"]),
            strength_index=int(row["strength_index"]),
        )
        # strength_index is part of the dedup key: under the universal-trigger
        # attack the docs (hence general_val_bpb) vary per k, so collapsing it
        # would make a k silently lose its row (review C2).
        key = (
            combo.target_model,
            combo.method_name,
            combo.concept_id,
            combo.coefficient_str,
            combo.adversarial_method,
            combo.adversarial_strength,
            combo.strength_index,
        )
        if key not in seen:
            seen.add(key)
            combos.append(combo)
    return combos


def _build_combo_intervention(
    *,
    combo: _InterventionCombo,
    method: Any,
    concept: Concept,
    source_paths: SourcePaths,
    diffmean_layer: int,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    tokenizer: Any,
    counter_direction_family: str,
    apply_max_act_scale: bool,
    adv_paths: Any | None = None,
    model: Any | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    source_validation: BpbAdvSourceValidation | None = None,
) -> Intervention:
    """Build the steered (+ optional additive counter) intervention for a combo.

    Mirrors ``ModelBackedDualCompletionScorer._steered_intervention`` but seeds
    the RNG deterministically from ``(method, concept, coefficient)`` rather than
    a ``core_prefix_id`` — the general-val intervention is a per-(method, concept,
    coefficient) representative, independent of any specific prefix.

    ``adv_paths`` is threaded into ``build_intervention`` so fair precomputed-bag
    methods can load universal bags keyed by ``(concept, mode, reg_lambda)`` only.
    Roofline precomputed-bag methods use the separate per-document path in
    ``_score_roofline_combo`` rather than this representative concept builder.

    ``model`` (the loaded target model) is forwarded to
    ``build_intervention`` -- REQUIRED there for every ``baseline == "ours"``
    method. Live methods use its authoritative checkpoint vocabulary for
    source-level restriction; fair precomputed-bag methods use it to verify the
    loaded bag's derive-time checkpoint V.
    """

    rng = random.Random(
        hashlib.sha256(
            f"general_val|{combo.method_name}|{combo.concept_id}|{combo.coefficient_str}".encode()
        ).digest()
    )
    coefficient = _coefficient_from_str(combo.coefficient_str)
    intervention_kwargs: dict[str, Any] = {
        "method": method,
        "concept": concept,
        "coefficient": coefficient,
        "source_paths": source_paths,
        "diffmean_layer": diffmean_layer,
        "target_model_meta": target_model_meta,
        "inactive_sampler": inactive_sampler,
        "tokenizer_nanochat": tokenizer,
        "rng": rng,
        "adv_paths": adv_paths,
        "core_prefix_id": None,
        "apply_max_act_scale": apply_max_act_scale,
        "model": model,
    }
    if cluster_contract is not None:
        intervention_kwargs["cluster_contract"] = cluster_contract
    if source_validation is not None and method.baseline in (
        BaselineKind.DIFFMEAN,
        BaselineKind.LINEAR_PROBE,
    ):
        intervention_kwargs["direction_payload"] = source_validation.direction_artifact_bytes(
            family=method.baseline.value,
            target=combo.target_model,
            mode=method.mode.value,
            concept_id=concept.concept_id,
        )
    base = build_intervention(**intervention_kwargs)
    # Attach the additive counter via the SHARED builder so the general-val
    # intervention is byte-identical to the one the scorer and generation build.
    return apply_additive_counter(
        base,
        adversarial_method=combo.adversarial_method,
        adversarial_strength=combo.adversarial_strength,
        method=method,
        concept=concept,
        source_paths=source_paths,
        diffmean_layer=diffmean_layer,
        target=combo.target_model,
        counter_direction_family=counter_direction_family,
        apply_max_act_scale=apply_max_act_scale,
        source_validation=source_validation,
    )


def _roofline_family(combo: _InterventionCombo) -> PrecomputedBagFamily | None:
    family = precomputed_bag_method_family(combo.method_name)
    if family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES:
        return family
    return None


def _roofline_reg_lambda(combo: _InterventionCombo, family: PrecomputedBagFamily) -> float:
    if family is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN:
        return 0.0
    coefficient = _coefficient_from_str(combo.coefficient_str)
    if coefficient is None:
        raise AxBenchBpbAdvError(
            f"roofline general_val: method {combo.method_name!r} requires a numeric "
            f"reg_lambda coefficient, got coefficient_str={combo.coefficient_str!r}"
        )
    return float(coefficient)


def _resolve_roofline_signal_budget(
    *,
    target_bundle: TargetModelBundle,
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> GeneralValSignalBudget:
    """Resolve the generic-val roofline signal budget from the encdec checkpoint.

    This intentionally reuses the same bucket and feature-count resolution as
    fair feature-DiffMean/GCG precomputed bags: the checkpoint's positive and
    negative selection-distribution means, unless explicitly overridden through
    ``cfg.ours_gcg.top_k_pos`` / ``top_k_neg``.
    """

    bucket_kwargs = {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        target_bundle.model,
        **bucket_kwargs,
    )
    expected_num_buckets = negative_bucket + 1
    if int(cfg.context_cache.num_buckets) != expected_num_buckets:
        raise AxBenchBpbAdvError(
            "roofline general_val: context_cache.num_buckets must match the "
            "encdec checkpoint bucket count before real output-window signals "
            f"can be discretized (cfg={cfg.context_cache.num_buckets}, "
            f"checkpoint={expected_num_buckets})"
        )
    k_pos, k_neg = resolve_k(
        target_bundle.model,
        cfg.ours_gcg,
        **bucket_kwargs,
    )
    return GeneralValSignalBudget(
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=positive_bucket,
        negative_bucket=negative_bucket,
    )


def _assert_roofline_bag_matches_checkpoint_vocab(
    metadata: Mapping[str, str],
    *,
    model: Any,
    bag: FeatureConditioning,
    doc: _GeneralValDoc,
    family: PrecomputedBagFamily,
    reg_lambda: float,
    path: Any,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    """Fail loudly if a persisted roofline general-val bag's stamped ``V`` disagrees with ``model``'s.

    The roofline-general-val sibling of
    :func:`steering.eval.axbench_bpb.intervention._assert_bag_matches_checkpoint_vocab`.
    The on-disk roofline-bag cache keys ONLY on ``(sample_id, family, reg_lambda)`` under an
    operator-chosen ``run_tag`` — no checkpoint/``V`` component — so reusing a ``run_tag``
    across checkpoints with a DIFFERENT compacted vocabulary would otherwise silently serve a
    bag conditioned on feature ids the currently-loaded checkpoint never saw (some may even
    be out of its embedding table's range). ``_write_roofline_bag`` STAMPS the derive-time
    fingerprint (via ``write_ours_gcg_bag``); this is the matching read-side check the fair
    path already has but the roofline read lacked —
    :func:`validate_general_val_bag_metadata` deliberately omits the fingerprint.

    Semantics mirror the fair path exactly (match / mismatch-raise /
    missing-when-compacted-raise / uncompacted-passes):

    * stamped == loaded — load unchanged.
    * stamped != loaded — RAISE (a run_tag reused across a different ``V``).
    * no stamp + COMPACTED checkpoint — RAISE: a bag written before stamping existed cannot
      prove it was derived for this ``V``.
    * no stamp + LEGACY (uncompacted) checkpoint — load unchanged: every pre-#695 bag was
      implicitly derived against the full vocabulary a legacy checkpoint has (byte-identity).
    """
    loaded_fingerprint = checkpoint_vocab_fingerprint(model)
    stamped_fingerprint = metadata.get(VOCAB_FINGERPRINT_META_KEY)
    if stamped_fingerprint is None:
        if loaded_fingerprint != UNCOMPACTED_VOCAB_FINGERPRINT:
            raise AxBenchBpbAdvError(
                f"roofline general_val: persisted bag at {path} for sample_id={doc.sample_id!r} "
                f"family={family.value!r} reg_lambda={reg_lambda!r} carries no "
                f"{VOCAB_FINGERPRINT_META_KEY!r} metadata (written before checkpoint-V stamping "
                f"existed), and the loaded checkpoint is COMPACTED (V-fingerprint="
                f"{loaded_fingerprint!r}). Cannot prove this bag was derived for this "
                "checkpoint's V — a bag derived against a different (or unmasked) vocabulary "
                "would silently condition on the wrong feature ids. Re-derive the bag under this "
                "checkpoint (clear general_val_bags/ or score under a fresh run_tag)."
            )
    elif stamped_fingerprint != loaded_fingerprint:
        raise AxBenchBpbAdvError(
            f"roofline general_val: persisted bag at {path} for sample_id={doc.sample_id!r} "
            f"family={family.value!r} reg_lambda={reg_lambda!r} was derived against "
            f"checkpoint V-fingerprint={stamped_fingerprint!r}, but the loaded checkpoint's "
            f"V-fingerprint={loaded_fingerprint!r}. The on-disk bag cache key carries no "
            "checkpoint/V component, so reusing a run_tag across checkpoints with a different "
            "V would otherwise silently serve a WRONG bag. Use a fresh run_tag for this "
            "checkpoint, or clear the stale general_val_bags/ cache."
        )
    stamped_contract = metadata.get(FEATURE_CONTRACT_FINGERPRINT_META_KEY)
    if cluster_contract is None:
        if stamped_contract is not None:
            raise AxBenchBpbAdvError(
                f"unclustered general_val reader rejects a cluster-stamped roofline bag at {path}"
            )
        return
    if stamped_contract != cluster_contract.feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            "roofline general_val bag feature-contract fingerprint mismatch: "
            f"expected={cluster_contract.feature_contract_fingerprint}, "
            f"actual={stamped_contract!r}, path={path}"
        )
    ids = t.tensor(list(bag.feature_ids()), dtype=t.int64)
    try:
        cluster_contract.consolidation.assert_representatives_only(
            ids,
            surface="roofline general_val persisted bag",
        )
        cluster_contract.feature_mask.assert_no_masked(ids)
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"roofline general_val persisted bag violates clustered target semantics: {exc}"
        ) from exc
    feature_vocab = cluster_contract.checkpoint.feature_vocab
    if feature_vocab is None:
        raise AxBenchBpbAdvError(
            "clustered roofline general_val requires a persistent checkpoint FeatureVocab"
        )
    checkpoint_vocab = set(feature_vocab.original_ids)
    invalid = sorted({int(value) for value in ids.tolist()} - checkpoint_vocab)
    if invalid:
        raise AxBenchBpbAdvError(
            "roofline general_val persisted bag contains representative IDs "
            f"outside the exact checkpoint vocabulary: {invalid[:20]}"
        )


def _load_persisted_roofline_bag(
    *,
    paths: AxBenchBpbAdvPaths,
    doc: _GeneralValDoc,
    family: PrecomputedBagFamily,
    reg_lambda: float,
    corpus_signature: str,
    ours: OursGcgConfig,
    signal_budget: GeneralValSignalBudget,
    config_metadata: dict[str, str],
    model: Any,
    warm_start_caps: GeneralValWarmStartCaps | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureConditioning | None:
    path = paths.general_val_bag(doc.sample_id, family, reg_lambda)
    if not path.exists():
        return None
    expected_contract_fingerprint = (
        None if cluster_contract is None else cluster_contract.feature_contract_fingerprint
    )
    bag, metadata = read_ours_gcg_bag(
        path,
        expected_feature_contract_fingerprint=expected_contract_fingerprint,
    )
    validate_general_val_bag_metadata(
        metadata=metadata,
        sample_id=doc.sample_id,
        family=family.value,
        reg_lambda=reg_lambda,
        corpus_signature=corpus_signature,
        ours=ours,
        signal_budget=signal_budget,
        config_metadata=config_metadata,
        warm_start_caps=warm_start_caps,
        cluster_contract=cluster_contract,
    )
    # Defense-in-depth (#695 PR-1 review): the identity validation above OMITS the
    # derive-time checkpoint V-fingerprint, so verify it here against the loaded checkpoint
    # — the read-side match to the stamp _write_roofline_bag records, mirroring the fair
    # path's _assert_bag_matches_checkpoint_vocab. A no-op on the uncompacted path
    # (feature_vocab is None both at derive and read).
    conditioning = FeatureConditioning.from_dict(bag)
    _assert_roofline_bag_matches_checkpoint_vocab(
        metadata,
        model=model,
        bag=conditioning,
        doc=doc,
        family=family,
        reg_lambda=reg_lambda,
        path=path,
        cluster_contract=cluster_contract,
    )
    logger.info(
        "roofline general_val: reusing persisted bag sample_id={} family={} reg_lambda={} -> {}",
        doc.sample_id,
        family.value,
        reg_lambda,
        path,
    )
    return conditioning


def _write_roofline_bag(
    *,
    paths: AxBenchBpbAdvPaths,
    doc: _GeneralValDoc,
    family: PrecomputedBagFamily,
    reg_lambda: float,
    bag: FeatureConditioning,
    metadata: dict[str, str],
    vocab_fingerprint: str,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    path = paths.general_val_bag(doc.sample_id, family, reg_lambda)
    if cluster_contract is not None:
        ids = t.tensor(list(bag.feature_ids()), dtype=t.int64)
        cluster_contract.consolidation.assert_representatives_only(
            ids,
            surface="roofline general_val bag writer",
        )
        cluster_contract.feature_mask.assert_no_masked(ids)
    write_ours_gcg_bag(path, bag, metadata=metadata, vocab_fingerprint=vocab_fingerprint)
    logger.info(
        "roofline general_val: wrote bag sample_id={} family={} reg_lambda={} n_features={} -> {}",
        doc.sample_id,
        family.value,
        reg_lambda,
        len(bag),
        path,
    )


def _roofline_bag_for_doc(
    *,
    combo: _InterventionCombo,
    family: PrecomputedBagFamily,
    doc: _GeneralValDoc,
    segments: tuple[ContextFeatureSegment, ...],
    roofline_state: _RooflineGeneralValState,
    target_bundle: TargetModelBundle,
    paths: AxBenchBpbAdvPaths,
    tokenizer: Any,
    device: t.device,
    corpus_signature: str,
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureConditioning:
    reg_lambda = _roofline_reg_lambda(combo, family)
    warm = roofline_state.real_bags.get(doc.sample_id)
    if warm is None:
        raise AxBenchBpbAdvError(
            "roofline general_val: missing real output-window warm bag for doc "
            f"{doc.sample_id!r}; it should have been derived during encdec "
            "feature extraction before Gemma offload"
        )
    warm_start_caps: GeneralValWarmStartCaps | None = None
    if family is PrecomputedBagFamily.ROOFLINE_GCG:
        gcg_kwargs = {} if cluster_contract is None else {"cluster_contract": cluster_contract}
        gcg_config = general_val_doc_gcg_config(
            target_bundle.model,
            warm,
            reg_lambda=reg_lambda,
            ours=cfg.ours_gcg,
            **gcg_kwargs,
        )
        warm_start_caps = GeneralValWarmStartCaps(
            max_k_pos=int(gcg_config.max_k_pos),
            max_k_neg=int(gcg_config.max_k_neg),
        )
    persisted = _load_persisted_roofline_bag(
        paths=paths,
        doc=doc,
        family=family,
        reg_lambda=reg_lambda,
        corpus_signature=corpus_signature,
        ours=cfg.ours_gcg,
        signal_budget=roofline_state.signal_budget,
        config_metadata=roofline_state.bag_config_metadata,
        model=target_bundle.model,
        warm_start_caps=warm_start_caps,
        cluster_contract=cluster_contract,
    )
    if persisted is not None:
        record_counter(
            "axbench_bpb_adv.general_val.roofline_bag_cache_hits",
            1,
            family=family.value,
        )
        return persisted
    record_counter(
        "axbench_bpb_adv.general_val.roofline_bag_cache_misses",
        1,
        family=family.value,
    )

    member_kwargs = {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    member = build_general_val_doc_member(
        doc,
        segments=segments,
        tokenizer=tokenizer,
        output_bag=warm,
        max_seq_len=target_bundle.max_seq_len,
        device=device,
        allowed_ids=feature_vocab_allowed_ids(target_bundle.model),
        **member_kwargs,
    )
    if family is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN:
        bag = warm
        objective_metadata: dict[str, str] = {}
    elif family is PrecomputedBagFamily.ROOFLINE_GCG:
        objective_kwargs = (
            {} if cluster_contract is None else {"cluster_contract": cluster_contract}
        )
        warm_objective = penalized_lm_objective_value(
            target_bundle.model,
            member,
            warm,
            reg_lambda=reg_lambda,
            ours=cfg.ours_gcg,
            device=device,
            **objective_kwargs,
        )
        bag = optimize_general_val_doc_bag(
            target_bundle.model,
            member,
            warm=warm,
            reg_lambda=reg_lambda,
            ours=cfg.ours_gcg,
            device=device,
            **objective_kwargs,
        )
        optimized_objective = penalized_lm_objective_value(
            target_bundle.model,
            member,
            bag,
            reg_lambda=reg_lambda,
            ours=cfg.ours_gcg,
            device=device,
            **objective_kwargs,
        )
        objective_metadata = {
            "warm_penalized_lm_objective": repr(float(warm_objective)),
            "optimized_penalized_lm_objective": repr(float(optimized_objective)),
        }
    else:
        raise AxBenchBpbAdvError(
            f"roofline general_val: unsupported roofline bag family {family.value!r}"
        )
    metadata = {
        **general_val_bag_identity_metadata(
            sample_id=doc.sample_id,
            family=family.value,
            reg_lambda=reg_lambda,
            corpus_signature=corpus_signature,
            ours=cfg.ours_gcg,
            signal_budget=roofline_state.signal_budget,
            config_metadata=roofline_state.bag_config_metadata,
            warm_start_caps=warm_start_caps,
            cluster_contract=cluster_contract,
        ),
        "method_name": combo.method_name,
        "coefficient_str": combo.coefficient_str,
        "n_features": str(len(bag)),
        **objective_metadata,
    }
    _write_roofline_bag(
        paths=paths,
        doc=doc,
        family=family,
        reg_lambda=reg_lambda,
        bag=bag,
        metadata=metadata,
        vocab_fingerprint=checkpoint_vocab_fingerprint(target_bundle.model),
        cluster_contract=cluster_contract,
    )
    record_counter(
        "axbench_bpb_adv.general_val.roofline_bag_writes",
        1,
        family=family.value,
    )
    return bag


def _score_roofline_combo(
    *,
    combo: _InterventionCombo,
    family: PrecomputedBagFamily,
    concept: Concept,
    method: Any,
    docs: list[_GeneralValDoc],
    samples_by_id: dict[str, ScoringInputs],
    segments_by_sample: dict[str, tuple[ContextFeatureSegment, ...]],
    roofline_state: _RooflineGeneralValState,
    target_bundle: TargetModelBundle,
    paths: AxBenchBpbAdvPaths,
    source_paths: SourcePaths,
    tokenizer: Any,
    device: t.device,
    cfg: AxBenchBpbAdvConfig,
    counter_direction_family: str,
    apply_max_act_scale: bool,
    corpus_signature: str,
    lora: LoraParams | None,
    cluster_contract: ClusterFeatureContract | None = None,
    source_validation: BpbAdvSourceValidation | None = None,
) -> list[Any]:
    if combo.target_model != "encdec":
        raise AxBenchBpbAdvError(
            "roofline general_val supports only encdec targets; got "
            f"target_model={combo.target_model!r} for method={combo.method_name!r}"
        )
    additive_counter = resolve_additive_counter(
        adversarial_method=combo.adversarial_method,
        adversarial_strength=combo.adversarial_strength,
        method=method,
        concept=concept,
        source_paths=source_paths,
        diffmean_layer=cfg.diffmean_and_probe_layer,
        target=combo.target_model,
        counter_direction_family=counter_direction_family,
        apply_max_act_scale=apply_max_act_scale,
        source_validation=source_validation,
    )
    rows: list[Any] = []
    for doc in docs:
        sample = samples_by_id.get(doc.sample_id)
        segments = segments_by_sample.get(doc.sample_id)
        if sample is None or segments is None:
            raise AxBenchBpbAdvError(
                "roofline general_val: doc "
                f"{doc.sample_id!r} is missing from the encdec scored sample set"
            )
        bag = _roofline_bag_for_doc(
            combo=combo,
            family=family,
            doc=doc,
            segments=segments,
            roofline_state=roofline_state,
            target_bundle=target_bundle,
            paths=paths,
            tokenizer=tokenizer,
            device=device,
            corpus_signature=corpus_signature,
            cfg=cfg,
            cluster_contract=cluster_contract,
        )
        if cluster_contract is not None:
            bag_ids = t.tensor(list(bag.feature_ids()), dtype=t.int64)
            cluster_contract.consolidation.assert_representatives_only(
                bag_ids,
                surface="roofline general_val score bag",
            )
            cluster_contract.feature_mask.assert_no_masked(bag_ids)
        intervention = attach_additive_counter(
            Intervention.ours(feature_conditions=dict(bag.entries)),
            additive_counter,
        )
        rows.extend(
            score_triple(
                samples=[sample],
                target_bundle=target_bundle,
                intervention=intervention,
                device=device,
                coefficient_str=combo.coefficient_str,
                method_name=combo.method_name,
                device_batch_size=cfg.device_batch_size,
                lora=lora,
            )
        )
    return rows


def _byte_weighted_bpb(rows: list[Any]) -> tuple[float, int]:
    """Corpus BPB = sum(nats) / (ln2 * sum(bytes)) = byte-weighted mean of the
    per-sample BPBs (each already nats/(ln2*bytes)). Returns (bpb, total_bytes)."""

    total_bytes = sum(int(row.n_output_bytes) for row in rows)
    if total_bytes <= 0:
        raise AxBenchBpbAdvError("general_val_bpb: zero scored bytes")
    weighted = sum(float(row.bpb_intervened) * int(row.n_output_bytes) for row in rows)
    return weighted / total_bytes, total_bytes


def _select_scored_signatures(
    signatures: list[str], *, max_interventions: int, seed: int
) -> set[str]:
    """Apply the optional ``max_interventions`` cap (0 = no cap).

    Deterministic, seeded subsample; logs the dropped count so the cap is never
    a silent truncation.
    """

    if max_interventions <= 0 or len(signatures) <= max_interventions:
        return set(signatures)
    ordered = sorted(signatures)
    rng = random.Random(seed)
    kept = set(rng.sample(ordered, max_interventions))
    logger.warning(
        "general_val_bpb: capping scored interventions at max_interventions={} "
        "(dropping {} of {} unique interventions; their general_val_bpb stays "
        "null).",
        max_interventions,
        len(signatures) - max_interventions,
        len(signatures),
    )
    return kept


def _corpus_signature(docs: list[_GeneralValDoc], *, prefix_tokens: int, output_tokens: int) -> str:
    """Self-describing per-target corpus identity: ``nd=<n>;nb=<bytes>;h=<32hex>``.

    The hash covers the ordered scored doc identities (sample_id, clean
    BOS-prefixed prefix ids, output ids, byte count) plus the slicing knobs.
    Attacked faster-GCG forks share the CLEAN signature: this describes the
    corpus, never the intervention.
    """
    hasher = hashlib.sha256()
    hasher.update(f"pt={int(prefix_tokens)};ot={int(output_tokens)}".encode())
    total_bytes = 0
    for doc in docs:
        n_bytes = int(doc.output_byte_weights.sum().item())
        total_bytes += n_bytes
        hasher.update(
            json.dumps(
                [doc.sample_id, list(doc.prefix_tokens), list(doc.output_tokens), n_bytes]
            ).encode()
        )
    return f"nd={len(docs)};nb={total_bytes};h={hasher.hexdigest()[:32]}"


def run_general_val_bpb_stage(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    cluster_contract: ClusterFeatureContract | None = None,
    stage_source: ClusteredStageSource | None = None,
) -> pa.Table:
    """Score the generic validation set under every distinct intervention and
    write the per-intervention ``general_val_bpb`` table.

    No-op stages (disabled) never reach here — the orchestrator gates the call on
    ``cfg.general_val_bpb.enabled``.
    """

    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=cluster_contract,
        surface="general-validation score stage",
    )
    source_validation = None if resolved_stage_source is None else resolved_stage_source.validation
    source_identity = (
        None if resolved_stage_source is None else resolved_stage_source.source_identity
    )
    direction_source_kwargs: dict[str, Any] = (
        {}
        if cluster_contract is None
        else {
            "cluster_contract": cluster_contract,
            "source_validation": source_validation,
        }
    )
    identity_kwargs: dict[str, Any] = (
        {} if source_identity is None else {"source_identity": source_identity}
    )
    gv = cfg.general_val_bpb
    with profile_region("axbench_bpb_adv.general_val.load_candidates"):
        candidate_snapshot = None
        if cluster_contract is None:
            candidates = load_candidate_table(paths.candidates_path())
        else:
            candidate_snapshot = load_clustered_candidate_snapshot(
                paths.candidates_path(),
                paths=paths,
                contract=cluster_contract,
                **identity_kwargs,
            )
            candidates = candidate_snapshot.table
    if not candidates.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError("expected candidates with CANDIDATES_SCHEMA")
    record_counter("axbench_bpb_adv.general_val.candidate_rows", candidates.num_rows)
    with profile_region(
        "axbench_bpb_adv.general_val.population_guard",
        rows=candidates.num_rows,
    ):
        candidates = require_table_in_evaluation_population(
            candidates,
            population=read_evaluation_population(
                paths.evaluation_population(),
                **(
                    {}
                    if cluster_contract is None
                    else {
                        "expected_feature_contract_fingerprint": (
                            cluster_contract.feature_contract_fingerprint
                        )
                    }
                ),
            ),
            label="general-val candidates",
        )
    with profile_region("axbench_bpb_adv.general_val.distinct_combos"):
        combos = _distinct_combos(candidates)
    if not combos:
        raise AxBenchBpbAdvError("no candidate interventions to score for general_val")
    record_counter("axbench_bpb_adv.general_val.distinct_combos", len(combos))
    targets = {combo.target_model for combo in combos}
    encdec_statistics: ExtractionStatistics | None = None
    if "encdec" in targets:
        # This stage's encoder-decoder path extracts fixed-token context
        # features. Validate the exact statistics snapshot before intervention
        # subsampling creates an RNG, RuntimeState loads either target, or any
        # output is written. Keep this parsed object and pass it through to the
        # extractor so the guarded file is not reopened later.
        encdec_statistics = load_statistics(cfg.feature_statistics_path)
        require_fixed_feature_dir_segment_mode(
            cfg.feature_dir,
            surface="AxBench-BPB-Adv general-val encoder-decoder extraction",
            statistics_segment_mode=encdec_statistics.metadata.segment_mode,
        )
    # One representative combo per signature carries the scored value. Compute this
    # before extraction so roofline real-output bags are derived only when a kept
    # roofline signature will actually be scored.
    combos_by_signature: dict[str, list[_InterventionCombo]] = {}
    for combo in combos:
        combos_by_signature.setdefault(_intervention_signature(combo), []).append(combo)
    kept_signatures = _select_scored_signatures(
        list(combos_by_signature),
        max_interventions=gv.max_interventions,
        seed=gv.seed,
    )
    record_counter(
        "axbench_bpb_adv.general_val.unique_signatures",
        len(combos_by_signature),
    )
    record_counter(
        "axbench_bpb_adv.general_val.kept_signatures",
        len(kept_signatures),
    )
    record_counter(
        "axbench_bpb_adv.general_val.dropped_signatures",
        len(combos_by_signature) - len(kept_signatures),
    )
    needs_roofline = any(
        _roofline_family(combos_by_signature[signature][0]) is not None
        for signature in kept_signatures
    )

    bpb_cfg = _bpb_config_from_adv_config(cfg)
    runtime_kwargs: dict[str, Any] = {
        "cfg": bpb_cfg,
        "ddp_rank": 0,
        "ddp_world_size": 1,
    }
    if cluster_contract is not None:
        runtime_kwargs["cluster_contract"] = cluster_contract
    runtime_state = RuntimeState(**runtime_kwargs)
    try:
        with profile_region(
            "axbench_bpb_adv.general_val.target_runtime_setup",
            n_targets=len(targets),
        ):
            target_bundles: dict[str, TargetModelBundle] = {}
            if "decoder_only" in targets:
                target_bundles["decoder_only"] = runtime_state.decoder_only()
            if "encdec" in targets:
                target_bundles["encdec"] = runtime_state.encdec()
        unknown = targets - set(target_bundles)
        if unknown:
            raise AxBenchBpbAdvError(f"unsupported target_model values: {sorted(unknown)}")
        roofline_state: _RooflineGeneralValState | None = None
        if needs_roofline:
            if "encdec" not in target_bundles:
                raise AxBenchBpbAdvError(
                    "general_val_bpb: roofline signatures require an encdec target bundle"
                )
            signal_budget = _resolve_roofline_signal_budget(
                target_bundle=target_bundles["encdec"],
                cfg=cfg,
                **({} if cluster_contract is None else {"cluster_contract": cluster_contract}),
            )
            roofline_state = _RooflineGeneralValState.for_budget(signal_budget)
            logger.info(
                "roofline general_val: resolved sparse real-signal budget "
                "k_pos={} k_neg={} positive_bucket={} negative_bucket={}",
                signal_budget.k_pos,
                signal_budget.k_neg,
                signal_budget.positive_bucket,
                signal_budget.negative_bucket,
            )
        tokenizer = next(iter(target_bundles.values())).tokenizer
        with profile_region("axbench_bpb_adv.general_val.build_token_bytes"):
            token_bytes = _build_token_bytes(tokenizer)
        device = t.device(runtime_state.target_device())

        with profile_region("axbench_bpb_adv.general_val.build_docs"):
            docs = build_general_val_docs(
                texts=iter_general_val_texts(),
                tokenizer=tokenizer,
                token_bytes=token_bytes,
                n_documents=gv.n_documents,
                prefix_tokens=gv.prefix_tokens,
                output_tokens=gv.output_tokens,
            )
        docs_corpus_signature = _corpus_signature(
            docs,
            prefix_tokens=gv.prefix_tokens,
            output_tokens=gv.output_tokens,
        )
        logger.info(
            "general_val_bpb: scoring {} val docs under {} unique intervention(s)",
            len(docs),
            len(combos),
        )

        # The universal-trigger artifact drives the DECODER-ONLY doc attack: the
        # per-concept trigger is substituted into each val doc's last-k so
        # general_val_bpb is measured UNDER the attack. Always present (the
        # candidate stage writes a schema-stable table even when empty).
        with profile_region("axbench_bpb_adv.general_val.load_universal_triggers"):
            universal_triggers = pq.read_table(paths.universal_triggers_path())
            assert_table_cluster_identity(
                universal_triggers,
                paths=paths,
                contract=cluster_contract,
                artifact_family="universal_triggers",
                candidate_snapshot=candidate_snapshot,
                **identity_kwargs,
            )
            trigger_index = _build_trigger_index(universal_triggers)
        record_counter("axbench_bpb_adv.general_val.trigger_rows", len(trigger_index))

        # encdec needs per-doc SAE features; decoder-only ignores segments.
        encdec_segments: dict[str, tuple[ContextFeatureSegment, ...]] = {}
        if "encdec" in targets:
            with profile_region(
                "axbench_bpb_adv.general_val.extract_encdec_segments",
                rows=len(docs),
            ):
                assert encdec_statistics is not None  # loaded with the encdec target above
                extraction_kwargs: dict[str, Any] = {
                    "runtime_state": runtime_state,
                    "cfg": cfg,
                    "statistics": encdec_statistics,
                    "roofline_state": roofline_state,
                    "tokenizer": tokenizer,
                    "device": device,
                    "allowed_ids": feature_vocab_allowed_ids(target_bundles["encdec"].model),
                }
                if cluster_contract is not None:
                    extraction_kwargs.update(
                        cluster_contract=cluster_contract,
                        corpus_signature=docs_corpus_signature,
                    )
                if cluster_contract is None:
                    # Preserve the exact legacy path for unclustered runs.
                    encdec_segments = _extract_encdec_segments(
                        docs,
                        **extraction_kwargs,
                    )
                else:
                    encdec_segments = _extract_with_target_memory_handoff(
                        runtime_state=runtime_state,
                        target_bundles=target_bundles,
                        extract=lambda: _extract_encdec_segments(
                            docs,
                            **extraction_kwargs,
                        ),
                    )
            record_counter(
                "axbench_bpb_adv.general_val.encdec_docs_with_segments",
                len(encdec_segments),
            )
            record_counter(
                "axbench_bpb_adv.general_val.encdec_segments",
                sum(len(value) for value in encdec_segments.values()),
            )
            record_counter(
                "axbench_bpb_adv.general_val.encdec_features",
                sum(
                    len(segment.feature_ids)
                    for segments in encdec_segments.values()
                    for segment in segments
                ),
            )
        # CLEAN (un-attacked) samples per target. Used for k=0 controls, additive
        # and base interventions, gepa (no trigger), and ALL encdec combos (the
        # encdec doc attack is deferred -- decoder-only scope). The decoder-only
        # faster_gcg k>0 attacked samples are rebuilt per-signature below.
        clean_samples_by_target: dict[str, list[ScoringInputs]] = {}
        if "decoder_only" in targets:
            clean_samples_by_target["decoder_only"] = [
                _doc_to_scoring_input(
                    doc,
                    segments=(),
                    max_seq_len=target_bundles["decoder_only"].max_seq_len,
                )
                for doc in docs
            ]
        if "encdec" in targets:
            clean_samples_by_target["encdec"] = [
                _doc_to_scoring_input(
                    doc,
                    segments=encdec_segments[doc.sample_id],
                    max_seq_len=target_bundles["encdec"].max_seq_len,
                )
                for doc in docs
                if doc.sample_id in encdec_segments
            ]
        # D2: a per-target corpus signature, stamped on every stage row for that
        # target. Computed from the SAME per-target scored-doc sets as
        # clean_samples_by_target above, so it agrees with what actually gets
        # scored (the cross-check below pins that agreement per signature).
        corpus_by_target: dict[str, str] = {}
        if "decoder_only" in targets:
            corpus_by_target["decoder_only"] = docs_corpus_signature
        if "encdec" in targets:
            encdec_docs = [doc for doc in docs if doc.sample_id in encdec_segments]
            corpus_by_target["encdec"] = _corpus_signature(
                encdec_docs, prefix_tokens=gv.prefix_tokens, output_tokens=gv.output_tokens
            )

        # Cross-target divergence backstop (D7): _extract_encdec_segments now
        # raises per-doc on an empty feature-segment tuple, so encdec and
        # decoder_only should always score the identical corpus when both
        # targets are active. This re-checks that invariant from the corpus
        # signatures themselves (rather than trusting the per-doc raise alone),
        # so a future refactor that reintroduces a per-doc skip still fails
        # loudly here instead of silently forking the two corpora (replaces the
        # #546 warn-only comparability caveat).
        if "decoder_only" in corpus_by_target and "encdec" in corpus_by_target:
            if corpus_by_target["decoder_only"] != corpus_by_target["encdec"]:
                raise AxBenchBpbAdvError(
                    "general_val_bpb: decoder_only and encdec scored different "
                    f"corpora ({corpus_by_target['decoder_only']!r} vs "
                    f"{corpus_by_target['encdec']!r}); comparability requires one "
                    "corpus per run (D7 backstop — _extract_encdec_segments should "
                    "have raised per-doc)"
                )

        source_paths = SourcePaths(Path(cfg.source_run_dir))
        with profile_region("axbench_bpb_adv.general_val.concepts_and_metadata"):
            concepts_by_id = (
                _read_concepts(source_paths)
                if source_validation is None
                else _concepts_from_entries(source_validation.concept_entries())
            )
            detect_kwargs = (
                {} if cluster_contract is None else {"cluster_contract": cluster_contract}
            )
            target_metas = {
                target: detect_target_model_metadata(
                    bundle.model,
                    bpb_cfg,
                    **detect_kwargs,
                )
                for target, bundle in target_bundles.items()
            }
        lora = (
            LoraParams(
                cfg=LoraConfig(**cfg.lora_adapter),
                layer=cfg.diffmean_and_probe_layer,
            )
            if cfg.lora_adapter
            else None
        )
        additive_cfg = get_additive_direction_config(cfg)
        counter_direction_family = (
            additive_cfg.counter_direction_family if additive_cfg is not None else "diffmean"
        )
        inactive_sampler = runtime_state.inactive_sampler() if "encdec" in targets else None

        with profile_region("axbench_bpb_adv.general_val.direction_scale_preflight"):
            _preflight_direction_scales_for_rows(
                candidate_rows=[
                    {
                        "method_name": combos_by_signature[s][0].method_name,
                        "concept_id": combos_by_signature[s][0].concept_id,
                        "target_model": combos_by_signature[s][0].target_model,
                        "adversarial_method": combos_by_signature[s][0].adversarial_method,
                        "adversarial_strength": (combos_by_signature[s][0].adversarial_strength),
                    }
                    for s in sorted(kept_signatures)
                ],
                concepts_by_id=concepts_by_id,
                source_paths=source_paths,
                expected_layer=cfg.diffmean_and_probe_layer,
                counter_direction_family=counter_direction_family,
                source_validation=source_validation,
                stage="general_val",
            )

        encdec_samples_by_id = {
            sample.sample_id: sample for sample in clean_samples_by_target.get("encdec", [])
        }
        signature_value: dict[str, tuple[float, int, int]] = {}
        signature_iterable = sorted(kept_signatures)
        for signature in signature_iterable:
            rep = combos_by_signature[signature][0]
            concept = concepts_by_id.get(rep.concept_id)
            if concept is None:
                raise AxBenchBpbAdvError(
                    f"source concepts do not contain concept_id={rep.concept_id}"
                )
            method = get_method_by_name(rep.method_name)
            roofline_family = _roofline_family(rep)
            if roofline_family is not None:
                if roofline_state is None:
                    raise AxBenchBpbAdvError(
                        "internal error: roofline signature reached scoring "
                        "without resolved roofline general-val state"
                    )
                with profile_region(
                    "axbench_bpb_adv.general_val.score_roofline_signature",
                    target_model=rep.target_model,
                    family=roofline_family.value,
                ):
                    rows = _score_roofline_combo(
                        combo=rep,
                        family=roofline_family,
                        concept=concept,
                        method=method,
                        docs=docs,
                        samples_by_id=encdec_samples_by_id,
                        segments_by_sample=encdec_segments,
                        roofline_state=roofline_state,
                        target_bundle=target_bundles[rep.target_model],
                        paths=paths,
                        source_paths=source_paths,
                        tokenizer=tokenizer,
                        device=device,
                        cfg=cfg,
                        counter_direction_family=counter_direction_family,
                        apply_max_act_scale=True,
                        corpus_signature=corpus_by_target[rep.target_model],
                        lora=lora,
                        **direction_source_kwargs,
                    )
            else:
                intervention = _build_combo_intervention(
                    combo=rep,
                    method=method,
                    concept=concept,
                    source_paths=source_paths,
                    diffmean_layer=cfg.diffmean_and_probe_layer,
                    target_model_meta=target_metas[rep.target_model],
                    inactive_sampler=inactive_sampler,
                    tokenizer=tokenizer,
                    counter_direction_family=counter_direction_family,
                    apply_max_act_scale=True,
                    adv_paths=paths,
                    model=target_bundles[rep.target_model].model,
                    **direction_source_kwargs,
                )
                # DECODER-ONLY attack: for a faster_gcg k>0 decoder-only rep,
                # substitute its universal trigger into each doc's last-k and rebuild
                # the scoring inputs; every other combo scores the clean docs.
                samples = _samples_for_combo(
                    rep,
                    docs=docs,
                    clean_samples=clean_samples_by_target[rep.target_model],
                    trigger_index=trigger_index,
                    tokenizer=tokenizer,
                    # The rep's own target bundle is always loaded (target_bundles is
                    # built from {combo.target_model}), so this is always valid and
                    # KeyErrors HERE (upstream) if that invariant is ever broken. Only
                    # the attacked decoder-only branch consumes it, where
                    # rep.target_model == "decoder_only".
                    decoder_max_seq_len=target_bundles[rep.target_model].max_seq_len,
                )
                with profile_region(
                    "axbench_bpb_adv.general_val.score_signature",
                    target_model=rep.target_model,
                    method_name=rep.method_name,
                    rows=len(samples),
                ):
                    rows = score_triple(
                        samples=samples,
                        target_bundle=target_bundles[rep.target_model],
                        intervention=intervention,
                        device=device,
                        coefficient_str=rep.coefficient_str,
                        method_name=rep.method_name,
                        device_batch_size=cfg.device_batch_size,
                        lora=lora,
                    )
            bpb, total_bytes = _byte_weighted_bpb(rows)
            signature_value[signature] = (bpb, len(rows), total_bytes)
            record_counter(
                "axbench_bpb_adv.general_val.scored_signature_docs",
                len(rows),
                target_model=rep.target_model,
            )
            record_counter(
                "axbench_bpb_adv.general_val.scored_signature_bytes",
                total_bytes,
                target_model=rep.target_model,
            )

        out_rows: list[dict[str, Any]] = []
        for combo in combos:
            signature = _intervention_signature(combo)
            scored = signature_value.get(signature)
            if scored is None:
                continue  # dropped by the max_interventions cap
            bpb, n_docs, n_bytes = scored
            expected = corpus_by_target[combo.target_model]
            if not expected.startswith(f"nd={n_docs};nb={n_bytes};"):
                raise AxBenchBpbAdvError(
                    f"general_val_bpb: scored row footprint nd={n_docs};nb={n_bytes} "
                    f"disagrees with the target corpus signature {expected!r} for "
                    f"signature {signature!r}; per-target doc sets must be stable "
                    "within one stage run"
                )
            out_rows.append(
                {
                    "target_model": combo.target_model,
                    "method_name": combo.method_name,
                    "mode": combo.mode,
                    "concept_id": combo.concept_id,
                    "coefficient_str": combo.coefficient_str,
                    "adversarial_method": combo.adversarial_method,
                    "adversarial_strength": combo.adversarial_strength,
                    "general_val_bpb": bpb,
                    "n_documents": n_docs,
                    "n_val_bytes": n_bytes,
                    "intervention_signature": signature,
                    "general_val_corpus": corpus_by_target[combo.target_model],
                }
            )
        table = pa.Table.from_pylist(out_rows, schema=GENERAL_VAL_BPB_SCHEMA)
        table = bind_clustered_table(
            table,
            paths=paths,
            contract=cluster_contract,
            artifact_family="general_val_bpb",
            candidate_snapshot=candidate_snapshot,
            **identity_kwargs,
        )
        with profile_region(
            "axbench_bpb_adv.general_val.write_output",
            rows=table.num_rows,
        ):
            if cluster_contract is None:
                atomic_write_parquet(table, paths.general_val_bpb_path())
            else:
                assert candidate_snapshot is not None
                write_clustered_table_atomic(
                    table,
                    paths.general_val_bpb_path(),
                    paths=paths,
                    contract=cluster_contract,
                    artifact_family="general_val_bpb",
                    candidate_snapshot=candidate_snapshot,
                    **identity_kwargs,
                )
        record_counter("axbench_bpb_adv.general_val.output_rows", table.num_rows)
        logger.info(
            "general_val_bpb: wrote {} rows ({} scored signatures) to {}",
            table.num_rows,
            len(signature_value),
            paths.general_val_bpb_path(),
        )
        return table
    finally:
        if cfg.offload_between_stages:
            runtime_state.offload_models()


__all__ = ["iter_general_val_texts", "run_general_val_bpb_stage"]
