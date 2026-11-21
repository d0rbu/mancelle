"""Stage 2: concept -> SAE feature mapping (spec 3.7).

Writes ``concept_features.safetensors`` (and optionally
``concept_features_neg.safetensors``) with four tensors + a
``metadata={"mode": ...}`` entry. Delta-ranked outputs also include
``metadata["extraction_tokenizer_id"]`` because their SAE extraction is
tokenizer-dependent.

Two modes, controlled by ``cfg.concept_feature_mode``:

- ``canonical`` (default): for each concept, slot 0 of
  ``feature_ids[concept_id]`` holds the Neuronpedia-mapped feature, the rest
  are ``-1``, and ``canonical_rank[concept_id]`` is ``0``. No GPU work, no
  synth data required.
- ``delta_ranked``: run base Gemma + SAE over the synth training
  data and compute
  ``percentile(concept_rich) - percentile(concept_avoiding)`` per
  SAE feature. Sort descending, store the top-``n_rank_store``
  feature IDs and their delta percentiles per concept. Tracks where
  the canonical (Neuronpedia-mapped) feature lands in the ranked
  list via ``canonical_rank``; if it's not in the top ``n_rank_store``,
  the rank is set to ``n_rank_store`` as a sentinel.

Schema (both modes):

- Row count is ``max(concept_id) + 1`` so original IDs remain direct indices.
  A filtered-out ID keeps an all-sentinel row rather than compacting later IDs.
- ``feature_ids``: int32, ``-1`` for unused slots and filtered-out rows.
- ``percentile_deltas``: float32, NaN for unused slots and filtered-out rows.
- ``canonical_feature_id``: int32, ``-1`` for filtered-out rows.
- ``canonical_rank``: int32, position of canonical in the ranked list;
  ``n_rank_store`` for filtered-out rows or a canonical feature outside top-N.

Both positive- and negative-mode synth corpora are now symmetric:
each concept owns its own ``concept_rich`` and ``concept_avoiding``
rows (positive mode emits a paired per-concept avoiding row per rich
row; the shared global-negatives pool and the ``concept_id == -1``
sentinel are gone). A single per-concept filter
(:func:`_concept_mode_filter`) therefore covers both modes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import torch as t
from loguru import logger
from safetensors.torch import save_file

from steering.conditional.percentile_table import magnitude_to_percentile
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
    validate_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.cluster_artifacts import (
    write_clustered_safetensors_once,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.extraction import (
    BosPrependedActivations,
    _bos_prepended_model_input,
    _infer_n_sae_features,
    _run_batched_extractor,
    _sae_hook_name,
    build_percentile_table_from_stats,
)
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
    get_extraction_tokenizer,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    Stage1ContinuationArtifactRole,
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
    Concept,
    TargetModelMetadata,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

_ModeFilter = Callable[
    [list[dict[str, Any]], int], tuple[list[str], list[str]],
]


def _concept_tensor_row_count(concepts: list[Concept]) -> int:
    """Return the direct-``concept_id`` tensor extent without compacting IDs."""

    if not concepts:
        raise AxBenchBaseError("Stage 2 requires a non-empty concept population")
    concept_ids = [concept.concept_id for concept in concepts]
    if concept_ids != sorted(set(concept_ids)):
        raise AxBenchBaseError(
            "Stage 2 concepts must have unique, increasing original concept IDs"
        )
    return concept_ids[-1] + 1


def _context_window(base_model: Any) -> int:
    """Return ``base_model.cfg.n_ctx``.

    Every SAE forward here now injects a BOS via
    :func:`~steering.eval.axbench_base.extraction._bos_prepended_model_input`,
    which requires a concrete ``n_ctx`` to front-clamp against -- there is
    no longer a "light stub, skip clamping" fallback. A missing ``cfg`` or
    ``cfg.n_ctx`` is a real config bug (wrong model type, failed load) and
    must raise loudly rather than silently degrade.

    Raises:
        AxBenchBaseError: ``base_model`` has no ``cfg.n_ctx``, or
            ``cfg.n_ctx`` is not positive.
    """
    cfg = getattr(base_model, "cfg", None)
    n_ctx = getattr(cfg, "n_ctx", None) if cfg is not None else None
    if n_ctx is None:
        raise AxBenchBaseError(
            "stage2 dense SAE max-pool: base_model has no cfg.n_ctx; "
            "cannot inject BOS or clamp."
        )
    n_ctx_int = int(n_ctx)
    if n_ctx_int <= 0:
        raise AxBenchBaseError(
            "stage2 dense SAE max-pool: base_model.cfg.n_ctx must be "
            f"positive, got {n_ctx_int}"
        )
    return n_ctx_int


def _concept_mode_filter(
    rows: list[dict[str, Any]], concept_id: int,
) -> tuple[list[str], list[str]]:
    """Filter synth rows into ``(concept_rich_texts, concept_avoiding_texts)``.

    Symmetric per-concept filter for BOTH positive (Stage 1b) and
    negative (Stage 1c) mode corpora: each side is gated on
    ``concept_id == cid``. Positive mode now emits a paired per-concept
    ``concept_avoiding`` row per rich row, so there is no longer any
    shared global-negatives pool (``concept_id == -1``) to fold in.

    - positives: ``concept_id == cid AND content_class == "concept_rich"``
    - negatives: ``concept_id == cid AND content_class == "concept_avoiding"``
    """
    pos = [
        r["output_text"]
        for r in rows
        if r["concept_id"] == concept_id and r["content_class"] == CONCEPT_RICH_LABEL
    ]
    neg = [
        r["output_text"]
        for r in rows
        if r["concept_id"] == concept_id and r["content_class"] == CONCEPT_AVOIDING_LABEL
    ]
    return pos, neg


def _raise_on_empty_synth_rows(
    pos_texts: list[str], neg_texts: list[str], concept_id: int,
) -> None:
    """RAISE if any synth-row ``output_text`` is empty / whitespace-only.

    Empty continuations are rejected at synth QC time — positive and negative
    modes skip a blank continuation (``stage1{b,c}.qc_empty_rejects``) before it
    can become a row, so it never reaches the corpus. An empty ``output_text``
    arriving at Stage 2 therefore means that early rejection FAILED: a serious
    upstream bug, not a routine degenerate row. Fail loudly here rather than
    silently drop it (which would hide the bug) or let
    :func:`_batched_dense_sae_max_pool`'s generic zero-token raise fire with a
    less specific message.
    """
    for label, texts in (
        ("concept_rich", pos_texts),
        ("concept_avoiding", neg_texts),
    ):
        n_empty = sum(1 for text in texts if not text.strip())
        if n_empty:
            raise AxBenchBaseError(
                f"Stage 2 concept {concept_id}: {n_empty} empty/whitespace "
                f"{label} synth row(s) reached the dense SAE pool. Empty rows must "
                "be rejected at synth QC time; an empty output_text this late "
                "means synth's early rejection failed — a serious upstream bug."
            )


def _batched_dense_sae_max_pool(
    texts: list[str],
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    batch_size: int = 8,
) -> t.Tensor:
    """Run base Gemma + SAE on ``texts`` and return a max-pooled per-feature matrix.

    Iterates ``texts`` in batches of ``batch_size``. Each non-empty text
    is tokenized and given exactly one injected leading BOS, front-clamped
    to ``base_model.cfg.n_ctx`` (see
    :func:`~steering.eval.axbench_base.extraction._bos_prepended_model_input`),
    then the batch is passed through :func:`_run_batched_extractor` at
    ``sae_layer``. Each returned ``(seq_len, n_features)`` tensor is
    max-pooled along the sequence dimension EXCLUDING the injected BOS
    row at position 0 -- the ``<bos>`` token is a large attention-sink
    outlier, and pooling it in would be out-of-distribution relative to
    the concept features this is compared against
    (:func:`steering.extract_features`'s ``real_acts = acts[1:]``
    convention).

    Every text must tokenize to at least one content token. A text that
    tokenizes to the empty list is a real data bug (a bare injected BOS is
    not valid SAE-forward input), so this RAISES ``AxBenchBaseError`` on it
    rather than silently emitting a zero-feature row that would understate the
    concept's features. The check happens BEFORE BOS injection, ahead of
    :func:`_bos_prepended_model_input` (which itself raises on zero content
    tokens). Callers holding data that may legitimately contain empty strings
    must filter them out before calling (the concept-mapping callers pass
    concept descriptions and non-empty synth-row texts, never empty strings).

    Callers pass ``target_model_meta.sae_layer`` for ``sae_layer``.

    Raises:
        AxBenchBaseError: a text tokenizes to zero tokens (an empty concept
            text -- refused, not zero-filled); ``base_model`` has no (positive)
            ``cfg.n_ctx`` when a non-empty text needs one to clamp against; the
            errors raised by :func:`_bos_prepended_model_input` (no BOS id,
            special-token pollution); the BOS + front-clamp leaves no content
            token for some text; the extractor returns a different number of
            tensors than token lists; or a row's activations aren't 1:1 with its
            token list (which would make the ``acts[1:]`` BOS-exclusion slice
            invalid).

    Returns a ``(n_texts, n_sae_features)`` float tensor.
    """
    n_features = _infer_n_sae_features(base_sae)
    if not texts:
        return t.zeros(0, n_features, dtype=t.float32)
    tokenizer = get_extraction_tokenizer(tokenizer_id)
    # Resolved lazily on the first non-empty text so an all-empty `texts`
    # list (or an all-empty batch) never requires base_model to expose
    # cfg.n_ctx -- matches the empty-text short-circuit below.
    n_ctx: int | None = None
    results: list[t.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        token_lists: list[list[int]] = []
        for text in batch_texts:
            # Concept texts must be non-empty: a zero-token concept text is a
            # real data bug (a bare injected BOS is not valid SAE input), NOT a
            # skippable zero-feature row. Raise loudly rather than silently
            # emitting a zeros() row that would understate the concept's features.
            raw_tokens = tokenizer.encode(text, add_special_tokens=False)
            if not raw_tokens:
                raise AxBenchBaseError(
                    "stage2 dense SAE max-pool: concept text tokenized to zero "
                    f"tokens (text: {text[:80]!r}). Concept texts must be "
                    "non-empty; refusing to emit a silent zero-feature row."
                )
            if n_ctx is None:
                n_ctx = _context_window(base_model)
            # Reuse raw_tokens (encoded_ids) so the helper does not re-tokenize;
            # it still runs the pollution guard + BOS-prepend + clamp.
            model_input = _bos_prepended_model_input(
                text,
                tokenizer=tokenizer,
                max_tokens=n_ctx,
                n_ctx=n_ctx,
                context="_batched_dense_sae_max_pool",
                encoded_ids=raw_tokens,
            )
            if len(model_input) < 2:
                raise AxBenchBaseError(
                    "stage2 dense SAE max-pool: after BOS + front-clamp "
                    "no content token remains for text (first 80 chars): "
                    f"{text[:80]!r} (n_ctx={n_ctx})."
                )
            logger.debug(
                f"stage2 dense SAE max-pool: BOS-injected input has "
                f"{len(model_input)} tokens (n_ctx={n_ctx})"
            )
            token_lists.append(model_input)
        # Every concept text produced a non-empty model_input (empties raise
        # above), so there is no empty-row bookkeeping: token_lists is 1:1 with
        # batch_texts and every row is scored.
        with profile_region(
            "stage2.batched_dense_sae_forward",
            rows=len(token_lists),
            sae_layer=sae_layer,
        ):
            acts_list = _run_batched_extractor(
                model=base_model,
                sae=base_sae,
                token_id_lists=token_lists,
                layer=sae_layer,
                hook_name=_sae_hook_name(base_sae),
            )
        if len(acts_list) != len(token_lists):
            raise AxBenchBaseError(
                "stage2 dense SAE max-pool: batched extractor returned "
                f"{len(acts_list)} activation tensors for "
                f"{len(token_lists)} token lists"
            )
        for acts, model_input in zip(acts_list, token_lists, strict=True):
            # from_forward asserts acts is 1:1 with model_input (no extractor
            # BOS injection); .content() is acts[1:] (BOS row 0 excluded).
            # Cast to fp32 at the SAE/downstream boundary: after
            # runtime_state.py:389 casts the SAE to bf16 for VRAM, these
            # activations are bf16 (~3 decimal digits) and downstream consumers
            # (delta_ranked mean, percentile lookup, Stage 3 DiffMean
            # accumulation) need fp32 to avoid silently degrading baselines.
            results.append(
                BosPrependedActivations.from_forward(acts, model_input)
                .content()
                .max(dim=0)
                .values.to(t.float32),  # (n_features,)
            )
        # Each batch has a different token length, so the CUDA caching
        # allocator accumulates a growing set of distinct block sizes across
        # hundreds of batches and never reuses the smaller ones -- confirmed
        # at c50 scale (~500 prompts): this is where the OOM traceback
        # actually fired (sae.encode(flat_acts) above). Nothing here holds a
        # live reference past this iteration once acts_list/token_lists are
        # dropped; release every batch, matching the project's established
        # per-iteration idiom (steering/eval/axbench_base/cuda_memory.py) --
        # its own docstring measured empty_cache() at ~0.7% overhead, cheap
        # enough to not need interval-gating.
        del acts_list, token_lists
        release_cuda_iteration_memory()
    return t.stack(results, dim=0)  # (n_texts, n_features), float32


def concept_feature_mapping_canonical(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[Path, Path | None]:
    """Write the trivial canonical concept-features file(s).

    For each surviving concept, slot 0 of
    ``feature_ids[concept.concept_id]`` holds ``concept.feature_index`` (the
    Neuronpedia-mapped feature), the matching canonical feature ID is stored in
    that row, and its canonical rank is 0. Rows for filtered-out original IDs
    remain all-sentinel.

    Writes ``concept_features.safetensors`` always, and
    ``concept_features_neg.safetensors`` as an identical copy if
    ``cfg.run_negative_mode`` is True.

    Cross-stage contract (audit slice 16, manifest §7 P3 #10):

    - Both files exist (when ``run_negative_mode=True``) for downstream
      Stage 4 symmetry: the positive-mode methods read
      ``concept_features.safetensors`` and the negative-mode methods
      read ``concept_features_neg.safetensors``. In canonical mode,
      both files are byte-identical because the feature_index→concept
      mapping is mode-agnostic (one Neuronpedia feature ID per
      concept).
    - The duplication is intentional. Refactoring Stage 4's negative-
      mode path to read the positive file in canonical mode would
      remove the duplication but requires symmetric handling on the
      delta-ranked path (which produces genuinely different positive vs
      negative files). Documented as a future cleanup in map P1-10.
    - Map P1-10 / P3 #10: in canonical mode the negative file is dead
      data on disk. Future cleanup options: (a) symlink
      ``concept_features_neg.safetensors`` →
      ``concept_features.safetensors`` when in canonical mode, or
      (b) refactor Stage 4 to share the same source-of-truth file
      across modes when ``concept_feature_mode == "canonical"``.
    """
    # Dispatcher guarantees run_tag is not None, but raise instead of
    # ``assert`` so the guard survives ``python -O`` (matches Stage 5
    # judge precedent — bare asserts are stripped under -O).
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "concept_feature_mapping_canonical: cfg.run_tag must be set "
            "(dispatcher should have populated it before calling)."
        )
    n = _concept_tensor_row_count(concepts)
    feature_ids = t.full((n, cfg.n_rank_store), -1, dtype=t.int32)
    percentile_deltas = t.full(
        (n, cfg.n_rank_store), float("nan"), dtype=t.float32,
    )
    canonical_feature_id = t.full((n,), -1, dtype=t.int32)
    canonical_rank = t.full((n,), cfg.n_rank_store, dtype=t.int32)

    if cluster_contract is not None:
        if cfg.concept_feature_mode != "canonical":
            raise AxBenchBaseError(
                "clustered Stage 2 requires concept_feature_mode=canonical"
            )
        concept_ids = t.tensor(
            [concept.feature_index for concept in concepts],
            dtype=t.int64,
        )
        cluster_contract.consolidation.assert_representatives_only(
            concept_ids,
            surface="clustered Stage 2 canonical concept IDs",
        )
        if cluster_contract.consolidation.feature_mask is not None:
            cluster_contract.consolidation.feature_mask.assert_no_masked(concept_ids)

    with profile_region("stage2.canonical_fill_tensors", rows=len(concepts)):
        for concept in concepts:
            row = concept.concept_id
            feature_ids[row, 0] = concept.feature_index
            canonical_feature_id[row] = concept.feature_index
            canonical_rank[row] = 0

    data: dict[str, t.Tensor] = {
        "feature_ids": feature_ids,
        "percentile_deltas": percentile_deltas,
        "canonical_feature_id": canonical_feature_id,
        "canonical_rank": canonical_rank,
    }
    metadata = {"mode": "canonical"}
    if cluster_contract is not None:
        # Track E2's intervention_dispatch port validates this value on consumption.
        metadata["feature_contract_fingerprint"] = cluster_contract.feature_contract_fingerprint
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    pos_path = output_paths.concept_features()
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_region("stage2.write_canonical_positive", rows=n):
        if cluster_contract is None:
            save_file(data, str(pos_path), metadata=metadata)
        else:
            write_clustered_safetensors_once(
                data,
                pos_path,
                metadata=metadata,
                surface="Stage 2 positive concept-feature mapping",
            )

    neg_path: Path | None = None
    if cfg.run_negative_mode:
        neg_path = output_paths.concept_features_neg()
        with profile_region("stage2.write_canonical_negative", rows=n):
            if cluster_contract is None:
                save_file(data, str(neg_path), metadata=metadata)
            else:
                write_clustered_safetensors_once(
                    data,
                    neg_path,
                    metadata=metadata,
                    surface="Stage 2 negative concept-feature mapping",
                )
    record_counter("stage2.concept_feature_rows", n)
    return pos_path, neg_path


def _load_synth_rows(
    path: Path,
    *,
    cfg: AxBenchBaseConfig,
    continuation_role: Stage1ContinuationArtifactRole,
) -> list[dict[str, Any]]:
    """Load a synth-training parquet as a list of row dicts.

    Uses pyarrow (matching the rest of the codebase) rather than pandas.
    The corpus is at most ``n_concepts x n_positives_per_concept``
    rows (~36k for 500 concepts x 72 positives), so list iteration is
    trivially fast.
    """
    if not path.exists():
        raise AxBenchBaseError(
            f"Synth training parquet not found at {path}; "
            "run Stage 1 before Stage 2"
        )
    with profile_region("stage2.load_synth_rows", path=path):
        table = pq.read_table(path)
    validate_extraction_tokenizer_metadata(
        path,
        expected_tokenizer_id=cfg.extraction_tokenizer_id,
        consumer="Stage 2",
        artifact_name="synth training parquet",
        schema=table.schema,
    )
    validate_stage1_continuation_contract(
        path,
        consumer="Stage 2",
        expected_role=continuation_role,
        schema=table.schema,
    )
    if continuation_role == "stage1b_positive":
        validate_positive_qc_contract(
            path,
            consumer="Stage 2",
            schema=table.schema,
            expected_accept_threshold=cfg.accept_threshold,
        )
        validate_positive_selection_contract(
            path,
            consumer="Stage 2",
            schema=table.schema,
            expected_cfg=cfg,
        )
    return table.to_pylist()


def _compute_ranked_file(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    rows: list[dict[str, Any]],
    filter_fn: _ModeFilter,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    out_path: Path,
    *,
    sae_layer: int,
) -> None:
    """Run the delta-ranked pipeline for one mode's worth of synth data.

    For each concept:

    1. Use ``filter_fn`` to split ``rows`` into ``concept_rich_texts``
       and ``concept_avoiding_texts``.
    2. Run Gemma + SAE max-pool on each side via
       :func:`_batched_dense_sae_max_pool`, mean-pool across texts.
    3. Convert means to percentiles via
       :func:`magnitude_to_percentile` on the pre-built
       ``percentile_table``.
    4. Sort by ``pos_pct - neg_pct`` descending, take the top
       ``cfg.n_rank_store`` entries.
    5. Locate the canonical (Neuronpedia-mapped) feature in the
       ranked list. If it's in the top N, record its rank; else
       set ``canonical_rank[concept_id] = cfg.n_rank_store`` as a sentinel.

    Callers pass ``target_model_meta.sae_layer`` for ``sae_layer``.
    Writes the resulting four tensors to ``out_path`` as a
    safetensors file tagged with mode and extraction tokenizer metadata.
    """
    n = _concept_tensor_row_count(concepts)
    feature_ids = t.full((n, cfg.n_rank_store), -1, dtype=t.int32)
    percentile_deltas = t.full(
        (n, cfg.n_rank_store), float("nan"), dtype=t.float32,
    )
    canonical_feature_id = t.full((n,), -1, dtype=t.int32)
    canonical_rank = t.full((n,), cfg.n_rank_store, dtype=t.int32)

    for concept in concepts:
        row = concept.concept_id
        canonical_feature_id[row] = concept.feature_index
        with profile_region(
            "stage2.filter_synth_rows_for_concept",
            concept_id=concept.concept_id,
            rows=len(rows),
        ):
            pos_texts, neg_texts = filter_fn(rows, concept.concept_id)
        # The filtered concept list is authoritative. Missing class rows now
        # prove a corrupt Stage 1 handoff; presence cannot silently become a
        # second population filter.
        _raise_on_empty_synth_rows(pos_texts, neg_texts, concept.concept_id)
        if not pos_texts or not neg_texts:
            missing_classes = [
                label
                for label, texts in (
                    (CONCEPT_RICH_LABEL, pos_texts),
                    (CONCEPT_AVOIDING_LABEL, neg_texts),
                )
                if not texts
            ]
            raise AxBenchBaseError(
                "Stage 2 authoritative filtered concept_id="
                f"{concept.concept_id} is missing "
                f"{', '.join(missing_classes)} rows in its synth corpus"
            )

        with profile_region(
            "stage2.pos_dense_sae_max_pool",
            concept_id=concept.concept_id,
            rows=len(pos_texts),
        ):
            pos_dense = _batched_dense_sae_max_pool(
                pos_texts, base_model, base_sae,
                sae_layer=sae_layer,
                tokenizer_id=cfg.extraction_tokenizer_id,
            )
        with profile_region(
            "stage2.neg_dense_sae_max_pool",
            concept_id=concept.concept_id,
            rows=len(neg_texts),
        ):
            neg_dense = _batched_dense_sae_max_pool(
                neg_texts,
                base_model,
                base_sae,
                sae_layer=sae_layer,
                tokenizer_id=cfg.extraction_tokenizer_id,
            )

        pos_mean = pos_dense.mean(dim=0)  # (n_features,)
        neg_mean = neg_dense.mean(dim=0)

        n_features = pos_mean.shape[0]
        all_fids = t.arange(n_features, dtype=t.int64)
        pos_pct = magnitude_to_percentile(all_fids, pos_mean, percentile_table)
        neg_pct = magnitude_to_percentile(all_fids, neg_mean, percentile_table)
        delta_pct = (pos_pct - neg_pct).to(t.float32)

        with profile_region(
            "stage2.rank_feature_deltas",
            concept_id=concept.concept_id,
            rows=int(delta_pct.numel()),
        ):
            sorted_idx = t.argsort(delta_pct, descending=True)
            top_n = min(cfg.n_rank_store, int(sorted_idx.numel()))
            top_fids = sorted_idx[:top_n].to(t.int32)
            top_deltas = delta_pct[sorted_idx[:top_n]].to(t.float32)

        feature_ids[row, :top_n] = top_fids
        percentile_deltas[row, :top_n] = top_deltas

        canonical_match = (top_fids == concept.feature_index).nonzero()
        if canonical_match.numel() > 0:
            canonical_rank[row] = int(canonical_match[0].item())
        else:
            canonical_rank[row] = cfg.n_rank_store

    data: dict[str, t.Tensor] = {
        "feature_ids": feature_ids,
        "percentile_deltas": percentile_deltas,
        "canonical_feature_id": canonical_feature_id,
        "canonical_rank": canonical_rank,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_region("stage2.write_delta_ranked_file", rows=n, path=out_path):
        save_file(
            data,
            str(out_path),
            metadata={
                "mode": "delta_ranked",
                EXTRACTION_TOKENIZER_METADATA_KEY: cfg.extraction_tokenizer_id,
            },
        )
    record_counter("stage2.delta_ranked_rows_written", n, path=out_path)


def concept_feature_mapping_delta_ranked(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    synth_training_positive_path: Path,
    synth_training_negative_path: Path | None,
    base_model: Any,
    base_sae: Any,
    target_model_meta: TargetModelMetadata,
) -> tuple[Path, Path | None]:
    """Run delta-ranked mapping for positive mode (and optionally negative mode).

    Loads the percentile table once (LRU-cached by
    ``feature_statistics_path``) so both mode runs share it.
    ``target_model_meta`` carries the SAE layer (via ``sae_layer``).
    """
    # Raise instead of ``assert`` so the guard survives ``python -O``
    # (bare asserts are stripped under -O).
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "concept_feature_mapping_delta_ranked: cfg.run_tag must be "
            "set (dispatcher should have populated it before calling)."
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    pos_rows = _load_synth_rows(
        synth_training_positive_path,
        cfg=cfg,
        continuation_role="stage1b_positive",
    )
    with profile_region("stage2.build_percentile_table"):
        percentile_table = build_percentile_table_from_stats(
            cfg.feature_statistics_path,
            n_sae_features=_infer_n_sae_features(base_sae),
        )
    sae_layer = target_model_meta.sae_layer
    pos_out = output_paths.concept_features()
    _compute_ranked_file(
        cfg,
        concepts,
        pos_rows,
        _concept_mode_filter,
        base_model,
        base_sae,
        percentile_table,
        pos_out,
        sae_layer=sae_layer,
    )

    neg_out: Path | None = None
    if cfg.run_negative_mode:
        if synth_training_negative_path is None:
            raise AxBenchBaseError(
                "run_negative_mode=True but synth_training_negative_path is None"
            )
        neg_rows = _load_synth_rows(
            synth_training_negative_path,
            cfg=cfg,
            continuation_role="stage1c_negative",
        )
        neg_out = output_paths.concept_features_neg()
        _compute_ranked_file(
            cfg,
            concepts,
            neg_rows,
            _concept_mode_filter,
            base_model,
            base_sae,
            percentile_table,
            neg_out,
            sae_layer=sae_layer,
        )
    return pos_out, neg_out
