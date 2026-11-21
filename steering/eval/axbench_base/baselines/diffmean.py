"""DiffMean training (Task 21, axbench_base Stage 3a).

Computes a residual-stream difference-of-means direction per concept:

    d = normalize( mean(h_pos) - mean(h_neg) )

where ``h_pos`` / ``h_neg`` are the mean continuation-token residual-
stream activations at layer ``cfg.layer_hooks.diffmean_and_probe_layer``,
captured from the target model on the prefix + output tokens of every
training row. "Continuation" means token positions that correspond to
``output_text``. The shared labeled-pair helper encodes
``prefix_text + output_text`` once, prepends one owned BOS, and uses decoded
token offsets to find the first token overlapping output text, including a BPE
token that crosses the character seam. The exact boundary must equal the
``n_prefix_nanochat_tokens`` committed by Stage 1; a mismatch rejects stale or
corrupt synth data rather than warning or guessing.

The residual-stream capture machinery and label-column constants are
shared with Task 22 (LinearProbe) via
``steering.eval.axbench_base.baselines._capture``.

Label column
------------
Both the positive-mode and negative-mode training-corpus parquets use
the same label column: ``content_class``.

  * ``content_class == "concept_rich"`` → DiffMean positive row
  * ``content_class == "concept_avoiding"`` → DiffMean negative row

Positive- and negative-mode corpora are now symmetric: each concept owns
its own ``concept_rich`` and ``concept_avoiding`` rows. Positive mode
(Stage 1b corpus) emits a paired per-concept ``concept_avoiding`` row per
rich row; the shared global-negatives pool and the ``concept_id == -1``
sentinel are gone. So in BOTH modes the per-concept filter is:

  positives: ``concept_id == c AND content_class == "concept_rich"``
  negatives: ``concept_id == c AND content_class == "concept_avoiding"``

With per-concept negatives, the positive class balance is now ~1:1 per
concept (was ~72 rich : 216 shared global negatives), so positives and
negatives are filtered identically by ``concept_id``.

Output
------
For each concept, writes a safetensors file to
``AxBenchBasePaths.diffmean_direction(target_model, mode, concept_id)``
with tensor key ``direction: (n_embd,)`` and text metadata
``{concept_id, mode, target_model, layer, n_pos, n_neg, max_activation}``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import save_file

from nanochat.tokenizer import get_tokenizer
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    validate_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines import _capture
from steering.eval.axbench_base.baselines.conditioning import EncoderConditioning
from steering.eval.axbench_base.cluster_artifacts import (
    write_clustered_safetensors_once,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    compute_direction_max_activation,
    encode_max_activation,
)
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.trainable_concepts import (
    concept_ids_present_in_rows,
    require_concept_population_coverage,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)

__all__ = ("train_diffmean_all",)


def _collect_row_vectors(
    rows: list[dict[str, Any]],
    *,
    model: nn.Module,
    layer: int,
    tokenizer: Any,
    side_label: str,
    concept_id: int,
    encoder_conditioning: EncoderConditioning | None = None,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """Run the canonical capture helper for every validated synth row.

    Returns ``(means, token_acts)`` — per-row ``(C,)`` continuation means (the
    fit inputs, unchanged) and per-row ``(T, C)`` continuation token matrices
    (consumed once per concept by the #1228 max_activation statistic, then
    freed with this function's caller frame).
    """
    means: list[t.Tensor] = []
    token_acts: list[t.Tensor] = []
    for row_idx, row in enumerate(rows):
        try:
            mean, acts = _capture._capture_per_row_continuation(
                model=model,
                encoder_conditioning=encoder_conditioning,
                layer=layer,
                tokenizer=tokenizer,
                prefix_text=row["prefix_text"],
                output_text=row["output_text"],
                n_prefix_nanochat_tokens=row["n_prefix_nanochat_tokens"],
            )
        except AxBenchBaseError as exc:
            raise AxBenchBaseError(
                f"DiffMean concept {concept_id} {side_label} row {row_idx} "
                f"violates the synth token contract: {exc}"
            ) from exc
        means.append(mean)
        token_acts.append(acts)
    return means, token_acts


def _train_diffmean_single(
    cfg: AxBenchBaseConfig,
    target_model: nn.Module,
    target_model_name: str,
    mode: SteeringMode,
    concept_id: int,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    clustered_metadata: dict[str, str] | None = None,
    encoder_conditioning: EncoderConditioning | None = None,
) -> Path:
    """Train the DiffMean direction for a single concept and save it.

    Returns the path the direction was written to.
    """
    layer = cfg.layer_hooks.diffmean_and_probe_layer

    pos_rows, neg_rows = _capture._filter_rows_for_concept(rows, concept_id)

    with profile_region(
        "stage3a.diffmean_collect_positive_vectors",
        concept_id=concept_id,
        rows=len(pos_rows),
    ):
        pos_vectors, pos_acts = _collect_row_vectors(
            pos_rows,
            model=target_model,
            layer=layer,
            tokenizer=tokenizer,
            encoder_conditioning=encoder_conditioning,
            side_label="positive",
            concept_id=concept_id,
        )
    with profile_region(
        "stage3a.diffmean_collect_negative_vectors",
        concept_id=concept_id,
        rows=len(neg_rows),
    ):
        neg_vectors, neg_acts = _collect_row_vectors(
            neg_rows,
            model=target_model,
            layer=layer,
            tokenizer=tokenizer,
            encoder_conditioning=encoder_conditioning,
            side_label="negative",
            concept_id=concept_id,
        )

    if not pos_vectors:
        raise AxBenchBaseError(
            f"Empty positives for concept {concept_id} in mode={mode}: "
            f"no rows with content_class=={_capture._POS_LABEL!r} survived filtering."
        )
    if not neg_vectors:
        raise AxBenchBaseError(
            f"Empty negatives for concept {concept_id} in mode={mode}: "
            f"no rows with content_class=={_capture._NEG_LABEL!r} survived filtering."
        )

    with profile_region("stage3a.diffmean_compute_direction", concept_id=concept_id):
        pos_stack = t.stack(pos_vectors, dim=0)  # (N_pos, C)
        neg_stack = t.stack(neg_vectors, dim=0)  # (N_neg, C)
        mean_pos = pos_stack.mean(dim=0)
        mean_neg = neg_stack.mean(dim=0)
        raw_direction = mean_pos - mean_neg

    if t.isnan(raw_direction).any():
        raise AxBenchBaseError(
            f"DiffMean direction contains NaN for concept {concept_id} "
            f"in mode={mode} (n_pos={len(pos_vectors)}, n_neg={len(neg_vectors)}). "
            "This usually means the target model produced NaN activations "
            "on one of the training rows; inspect the corpus for "
            "tokenization or content issues."
        )

    direction_norm = raw_direction.norm()
    if direction_norm.item() == 0.0:
        logger.warning(
            f"DiffMean concept {concept_id} mode={mode}: direction has zero "
            f"norm (mean_pos == mean_neg exactly). Saving zero vector; "
            "downstream coefficient-grid will evaluate this as a no-op."
        )
        direction = raw_direction
    else:
        direction = F.normalize(raw_direction, dim=-1)

    max_activation = compute_direction_max_activation(
        pos_acts + neg_acts,
        direction,
        surface=f"DiffMean concept {concept_id} mode={mode}",
    )

    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling train_diffmean_all",
        )
    out_path = paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
        target_model=target_model_name,
        mode=mode,
        concept_id=concept_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_region("stage3a.diffmean_save_direction", concept_id=concept_id):
        metadata = {
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": target_model_name,
            "layer": str(layer),
            "n_pos": str(len(pos_vectors)),
            "n_neg": str(len(neg_vectors)),
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(max_activation),
        }
        if encoder_conditioning is not None:
            # #1249: the direction was fit under THIS feature bag; the digest
            # travels with the artifact so a direction can never be silently
            # reused against a different bag.
            metadata["encdec_fit_bag_digest"] = encoder_conditioning.bag_digest
        if clustered_metadata is not None:
            metadata.update(clustered_metadata)
        tensors = {"direction": direction.contiguous()}
        if clustered_metadata is None:
            save_file(tensors, str(out_path), metadata=metadata)
        else:
            write_clustered_safetensors_once(
                tensors,
                out_path,
                metadata=metadata,
                surface=(
                    f"Stage 3 DiffMean {mode} concept {concept_id} direction"
                ),
            )
    logger.info(
        f"DiffMean concept {concept_id} mode={mode}: wrote direction "
        f"(n_pos={len(pos_vectors)}, n_neg={len(neg_vectors)}) to {out_path}"
    )
    return out_path


def train_diffmean_all(
    cfg: AxBenchBaseConfig,
    target_model: nn.Module,
    target_model_name: str,
    mode: SteeringMode,
    concepts: list[Concept],
    *,
    training_corpus_path: Path,
    feature_contract_fingerprint: str | None = None,
    encoder_conditioning_for: Callable[[int], EncoderConditioning] | None = None,
) -> list[Path]:
    """Train DiffMean directions for every concept in ``concepts``.

    Args:
        cfg: Axbench-base config; ``cfg.layer_hooks.diffmean_and_probe_layer``
            selects the residual-stream layer, and ``cfg.run_tag`` is used
            to resolve output paths.
        target_model: The decoder-only or encoder-decoder model to run
            forwards through. Must expose ``model.transformer.h[layer]``.
        target_model_name: ``"decoder_only"`` or ``"encdec"``; used as a
            path segment and saved into metadata.
        mode: ``"positive"`` or ``"negative"``; used as a path segment
            and saved into metadata.
        concepts: List of concepts to train directions for.
        training_corpus_path: Path to the synth training parquet
            (``synth_training_positive.parquet`` for positive mode,
            ``synth_training_negative.parquet`` for negative mode).

    Returns:
        List of output paths (one path per PRESENT concept, in ``concepts``
        order — concepts absent from the synth corpus are skipped (issue #508)).

    Raises:
        AxBenchBaseError: on empty positives/negatives for any concept,
            on NaN in any direction, or on ``cfg.run_tag is None``.
    """
    if not concepts:
        raise AxBenchBaseError(
            "train_diffmean_all called with empty concepts list",
        )

    with profile_region("stage3a.diffmean_load_training_rows"):
        if feature_contract_fingerprint is None:
            table = pq.read_table(training_corpus_path)
            clustered_corpus_sha256 = None
        else:
            table, clustered_corpus_sha256 = _capture._read_cluster_pinned_training_table(
                training_corpus_path
            )
        validate_extraction_tokenizer_metadata(
            training_corpus_path,
            expected_tokenizer_id=cfg.extraction_tokenizer_id,
            consumer="Stage 3a DiffMean",
            artifact_name="synth training parquet",
            schema=table.schema,
        )
        validate_stage1_continuation_contract(
            training_corpus_path,
            consumer="Stage 3a DiffMean",
            expected_role=(
                "stage1b_positive"
                if mode == SteeringMode.POSITIVE
                else "stage1c_negative"
            ),
            schema=table.schema,
        )
        if mode == SteeringMode.POSITIVE:
            validate_positive_qc_contract(
                training_corpus_path,
                consumer="Stage 3a DiffMean",
                schema=table.schema,
                expected_accept_threshold=cfg.accept_threshold,
            )
            validate_positive_selection_contract(
                training_corpus_path,
                consumer="Stage 3a DiffMean",
                schema=table.schema,
                expected_cfg=cfg,
            )
        rows = table.to_pylist()
    logger.info(
        f"DiffMean: loaded {len(rows)} training rows from {training_corpus_path} "
        f"for {len(concepts)} concepts (mode={mode}, target={target_model_name})"
    )
    present = concept_ids_present_in_rows(rows)
    require_concept_population_coverage(
        (concept.concept_id for concept in concepts),
        present,
        mode=mode,
        consumer="Stage 3a DiffMean",
    )
    tokenizer = get_tokenizer()
    if feature_contract_fingerprint is None:
        clustered_metadata = None
    else:
        # The clustered reader's tuple return contract always supplies this digest.
        if clustered_corpus_sha256 is None:  # pragma: no cover
            raise AxBenchBaseError(
                "Clustered DiffMean corpus digest was not resolved before training."
            )
        clustered_metadata = {
            "feature_contract_fingerprint": feature_contract_fingerprint,
            "source_corpus_sha256": clustered_corpus_sha256,
            "artifact_kind": "diffmean_direction_v1",
        }
    clustered_kwargs = (
        {} if clustered_metadata is None else {"clustered_metadata": clustered_metadata}
    )

    out_paths: list[Path] = []
    for concept in concepts:
        with profile_region(
            "stage3a.diffmean_train_single",
            concept_id=concept.concept_id,
            mode=mode,
            target_model=target_model_name,
        ):
            out_path = _train_diffmean_single(
                cfg=cfg,
                target_model=target_model,
                target_model_name=target_model_name,
                mode=mode,
                concept_id=concept.concept_id,
                rows=rows,
                tokenizer=tokenizer,
                encoder_conditioning=(
                    None
                    if encoder_conditioning_for is None
                    else encoder_conditioning_for(concept.concept_id)
                ),
                **clustered_kwargs,
            )
        out_paths.append(out_path)
    record_counter(
        "stage3a.diffmean_directions_written",
        len(out_paths),
        mode=mode,
        target_model=target_model_name,
    )
    return out_paths
