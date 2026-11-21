"""LinearProbe training (Task 22, axbench_base Stage 3a).

Fits an L1-regularised logistic regression on per-row mean
continuation-token residual-stream activations at layer
``cfg.layer_hooks.diffmean_and_probe_layer``, captured from the target
model on the prefix + output tokens of every training row. The
unit-normalised coefficient vector of the fitted probe is the saved
steering direction.

Shared with Task 21 (DiffMean)
------------------------------
Residual-stream capture, label-column constants, per-concept row
filtering, and the BOS-prepend tokenization logic are all shared with
``baselines.diffmean`` via ``baselines._capture``. See the DiffMean
module docstring for the full description of the label convention
(``content_class`` column, ``concept_rich`` vs ``concept_avoiding``)
and the per-concept negative filtering rule (both modes are now
symmetric; there is no global-negative ``concept_id == -1`` pool).

Probe config
------------
``sklearn.linear_model.LogisticRegression`` with:

  * ``l1_ratio=1.0``  — pure L1 penalty, equivalent to the legacy
    ``penalty="l1"`` kwarg. sklearn 1.8 deprecated the ``penalty=``
    keyword in favour of ``l1_ratio=`` (which generalises to elastic
    net), so we use the forward-compatible spelling. The saved
    metadata still records ``"penalty": "l1"`` for user-facing
    stability.
  * ``C=1.0``         — moderate L1 strength
  * ``solver="liblinear"``  — required for L1
  * ``max_iter=500``  — generous budget for the tiny n_samples we see
  * ``random_state=cfg.generation.master_seed``  — fully deterministic
  * ``class_weight=None``  — we deliberately do NOT rebalance classes.
    With per-concept negatives the positive-mode corpus is now ~1:1
    positive:negative per concept (was ~1:3 against the shared global
    pool), so the classes are already near-balanced and rebalancing is
    unnecessary; it would only distort the decision boundary away from
    the class-conditional means that dense L1 on residuals is trying to
    learn. (No logic change: ``class_weight`` stays ``None``.)

Output
------
For each concept, writes a safetensors file to
``AxBenchBasePaths.linear_probe_direction(target_model, mode, concept_id)``
with three tensor keys:

  * ``direction: (C,)`` — unit-norm ``coef_[0]`` (the steering vector)
  * ``coef: (1, C)``    — the raw ``coef_`` from ``LogisticRegression``
  * ``intercept: (1,)`` — the raw ``intercept_`` from the probe

and text metadata
``{concept_id, mode, target_model, layer, n_pos, n_neg, C, penalty,
max_activation}``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression

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

__all__ = ("LogisticRegression", "train_linear_probe_all")


# --- Probe hyperparameters (frozen — see module docstring) ---
# ``_PROBE_PENALTY`` is the user-facing label written to metadata; the
# sklearn 1.8+ API uses ``l1_ratio`` internally (``l1_ratio=1.0`` is
# equivalent to the legacy ``penalty='l1'``).
_PROBE_PENALTY: str = "l1"
_PROBE_L1_RATIO: float = 1.0
_PROBE_C: float = 1.0
_PROBE_MAX_ITER: int = 500
_PROBE_SOLVER: str = "liblinear"


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
                f"LinearProbe concept {concept_id} {side_label} row {row_idx} "
                f"violates the synth token contract: {exc}"
            ) from exc
        means.append(mean)
        token_acts.append(acts)
    return means, token_acts


def _train_linear_probe_single(
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
    """Train the LinearProbe direction for a single concept and save it.

    Returns the path the direction was written to.
    """
    layer = cfg.layer_hooks.diffmean_and_probe_layer

    pos_rows, neg_rows = _capture._filter_rows_for_concept(rows, concept_id)

    with profile_region(
        "stage3b.linear_probe_collect_positive_vectors",
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
        "stage3b.linear_probe_collect_negative_vectors",
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

    pos_stack = t.stack(pos_vectors, dim=0)  # (N_pos, C)
    neg_stack = t.stack(neg_vectors, dim=0)  # (N_neg, C)

    if t.isnan(pos_stack).any() or t.isnan(neg_stack).any():
        raise AxBenchBaseError(
            f"LinearProbe activations contain NaN for concept {concept_id} "
            f"in mode={mode} (n_pos={len(pos_vectors)}, n_neg={len(neg_vectors)}). "
            "This usually means the target model produced NaN activations "
            "on one of the training rows; inspect the corpus for "
            "tokenization or content issues."
        )

    # Stack features (positives then negatives) + labels; sklearn wants numpy.
    x_tensor = t.cat([pos_stack, neg_stack], dim=0)
    x_numpy = x_tensor.numpy().astype(np.float32)
    y_numpy = np.concatenate(
        [
            np.ones(len(pos_vectors), dtype=np.int64),
            np.zeros(len(neg_vectors), dtype=np.int64),
        ]
    )

    # Reference ``LogisticRegression`` by bare name so tests can
    # spy on / replace it via monkeypatch.setattr on the module dict.
    with profile_region("stage3b.linear_probe_fit", concept_id=concept_id):
        probe = LogisticRegression(
            l1_ratio=_PROBE_L1_RATIO,
            C=_PROBE_C,
            max_iter=_PROBE_MAX_ITER,
            solver=_PROBE_SOLVER,
            random_state=cfg.generation.master_seed,
        )
        probe.fit(x_numpy, y_numpy)

    coef_numpy = np.asarray(probe.coef_, dtype=np.float32)
    intercept_numpy = np.asarray(probe.intercept_, dtype=np.float32)
    coef_tensor = t.from_numpy(coef_numpy).contiguous()  # (1, C)
    intercept_tensor = t.from_numpy(intercept_numpy).contiguous()  # (1,)

    if t.isnan(coef_tensor).any() or t.isnan(intercept_tensor).any():
        raise AxBenchBaseError(
            f"LinearProbe fit produced NaN coefficients for concept "
            f"{concept_id} in mode={mode} (n_pos={len(pos_vectors)}, "
            f"n_neg={len(neg_vectors)}). The fit did not converge to a "
            "finite solution; inspect the activation scale."
        )

    # Clone so ``direction`` does not share storage with ``coef_tensor``
    # — safetensors refuses to write two tensors whose storage pointers
    # overlap (the normalised direction is a view of the raw coef row
    # otherwise, and safetensors would raise).
    raw_direction = coef_tensor[0].clone()  # (C,)
    direction_norm = raw_direction.norm()
    if direction_norm.item() == 0.0:
        logger.warning(
            f"LinearProbe concept {concept_id} mode={mode}: fitted coef has "
            f"zero norm (L1 drove every feature to zero). Saving zero vector; "
            "downstream coefficient-grid will evaluate this as a no-op."
        )
        direction = raw_direction
    else:
        direction = F.normalize(raw_direction, dim=-1)

    max_activation = compute_direction_max_activation(
        pos_acts + neg_acts,
        direction,
        surface=f"LinearProbe concept {concept_id} mode={mode}",
    )

    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling train_linear_probe_all",
        )
    out_path = paths.AxBenchBasePaths.from_config(cfg).linear_probe_direction(
        target_model=target_model_name,
        mode=mode,
        concept_id=concept_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_region("stage3b.linear_probe_save_direction", concept_id=concept_id):
        metadata = {
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": target_model_name,
            "layer": str(layer),
            "n_pos": str(len(pos_vectors)),
            "n_neg": str(len(neg_vectors)),
            "C": str(_PROBE_C),
            "penalty": _PROBE_PENALTY,
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(max_activation),
        }
        if encoder_conditioning is not None:
            # #1249: the direction was fit under THIS feature bag; the digest
            # travels with the artifact so a direction can never be silently
            # reused against a different bag.
            metadata["encdec_fit_bag_digest"] = encoder_conditioning.bag_digest
        if clustered_metadata is not None:
            metadata.update(clustered_metadata)
        tensors = {
            "direction": direction.contiguous(),
            "coef": coef_tensor,
            "intercept": intercept_tensor,
        }
        if clustered_metadata is None:
            save_file(tensors, str(out_path), metadata=metadata)
        else:
            write_clustered_safetensors_once(
                tensors,
                out_path,
                metadata=metadata,
                surface=(
                    f"Stage 3 linear-probe {mode} concept {concept_id} direction"
                ),
            )
    logger.info(
        f"LinearProbe concept {concept_id} mode={mode}: wrote direction "
        f"(n_pos={len(pos_vectors)}, n_neg={len(neg_vectors)}) to {out_path}"
    )
    return out_path


def train_linear_probe_all(
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
    """Train LinearProbe directions for every concept in ``concepts``.

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
            on NaN in any activation or fit result, or on
            ``cfg.run_tag is None``.
    """
    if not concepts:
        raise AxBenchBaseError(
            "train_linear_probe_all called with empty concepts list",
        )

    with profile_region("stage3b.linear_probe_load_training_rows"):
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
            consumer="Stage 3b LinearProbe",
            artifact_name="synth training parquet",
            schema=table.schema,
        )
        validate_stage1_continuation_contract(
            training_corpus_path,
            consumer="Stage 3b LinearProbe",
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
                consumer="Stage 3b LinearProbe",
                schema=table.schema,
                expected_accept_threshold=cfg.accept_threshold,
            )
            validate_positive_selection_contract(
                training_corpus_path,
                consumer="Stage 3b LinearProbe",
                schema=table.schema,
                expected_cfg=cfg,
            )
        rows = table.to_pylist()
    logger.info(
        f"LinearProbe: loaded {len(rows)} training rows from {training_corpus_path} "
        f"for {len(concepts)} concepts (mode={mode}, target={target_model_name})"
    )
    present = concept_ids_present_in_rows(rows)
    require_concept_population_coverage(
        (concept.concept_id for concept in concepts),
        present,
        mode=mode,
        consumer="Stage 3b LinearProbe",
    )
    tokenizer = get_tokenizer()
    if feature_contract_fingerprint is None:
        clustered_metadata = None
    else:
        # The clustered reader's tuple return contract always supplies this digest.
        if clustered_corpus_sha256 is None:  # pragma: no cover
            raise AxBenchBaseError(
                "Clustered LinearProbe corpus digest was not resolved before training."
            )
        clustered_metadata = {
            "feature_contract_fingerprint": feature_contract_fingerprint,
            "source_corpus_sha256": clustered_corpus_sha256,
            "artifact_kind": "linear_probe_direction_v1",
        }
    clustered_kwargs = (
        {} if clustered_metadata is None else {"clustered_metadata": clustered_metadata}
    )

    out_paths: list[Path] = []
    for concept in concepts:
        with profile_region(
            "stage3b.linear_probe_train_single",
            concept_id=concept.concept_id,
            mode=mode,
            target_model=target_model_name,
        ):
            out_path = _train_linear_probe_single(
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
        "stage3b.linear_probe_directions_written",
        len(out_paths),
        mode=mode,
        target_model=target_model_name,
    )
    return out_paths
