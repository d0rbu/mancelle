"""Teacher-forced BPB scoring kernel. Produces BpbScoreRow per sample."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from beartype import beartype
from loguru import logger

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.eval.axbench_base.profiler import record_counter
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.seg_map import (
    build_multi_segment_seg_map,
    project_segments_to_truncated_prefix,
)
from steering.eval.axbench_bpb.tokenization import truncate_prefix_left
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    BpbScoreRow,
    Intervention,
    LoraParams,
    ScoringInputs,
    TargetModel,
)
from steering.eval.axbench_bpb.vocab_restriction import (
    feature_vocab_allowed_ids,
    restrict_to_vocab,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


@dataclass(frozen=True)
class _DecoderPaddedInputs:
    """Padded (B, T_max-1) decoder inputs ready for a single forward pass.

    Built by `_pad_decoder_inputs`; consumed by `_score_microbatch` and
    `_extract_per_sample_bpb`. All tensors live on the target device.
    """
    dec_input: torch.Tensor       # (B, T_max-1) long, padded with token id 0
    dec_target: torch.Tensor      # (B, T_max-1) long, padded with -1 (model ignore_index)
    loss_mask: torch.Tensor       # (B, T_max-1) bool — True at output positions only
    n_full_prefix: list[int]
    n_output: list[int]
    truncated: list[bool]
    full_prefixes: list[list[int]]  # post-prompt-fold + post-truncation per sample
    T_max: int


@dataclass(frozen=True)
class _EncoderPaddedInputs:
    """Padded (B, n_seg_max, F_max) encoder inputs + (B, T_max-1) seg_map.

    Built by `_pad_encoder_inputs` for `target_name == "encdec"`; consumed
    only by the encdec branch of `_score_microbatch`.

    Relation to `steering.eval.multiseg_generator.MultisegInputs`: same four
    fields (enc_feature_ids/enc_magnitudes/enc_mask/seg_map) but different
    contract — `MultisegInputs` is built per single-row generation call with
    `(1, n_seg, F_max)` encoder shape and `(1, T)` seg_map (no shift), while
    this type is the batched microbatch shape `(B, n_seg_max, F_max)` with
    `(B, T_max-1)` seg_map (post-shift, padded with per-row generation_slot).
    Conflating the two would cause subtle bugs at the cross-attention scatter,
    so the types are deliberately separate.
    """
    enc_feature_ids: torch.Tensor  # (B, n_seg_max, F_max) int64
    enc_magnitudes: torch.Tensor   # (B, n_seg_max, F_max) int64
    enc_mask: torch.Tensor         # (B, n_seg_max, F_max) bool
    seg_map: torch.Tensor          # (B, T_max-1) long, padded with per-row generation_slot
    allowed_ids: torch.Tensor | None  # checkpoint's compacted vocabulary V; None if uncompacted


@beartype
def build_output_only_mask(
    *,
    n_full_prefix: PositiveInt,
    n_output: PositiveInt,
) -> torch.Tensor:
    """Return a (T-1,) bool mask where T = n_full_prefix + n_output.

    After the decoder shift-by-1, output-token *targets* occupy indices
    n_full_prefix - 1 .. T - 2. Prefix positions get False.
    """
    total = n_full_prefix + n_output
    mask = torch.zeros(total - 1, dtype=torch.bool)
    mask[n_full_prefix - 1:] = True
    return mask


def accumulate_bpb(
    loss_flat: torch.Tensor,
    mask: torch.Tensor,
    byte_weights_output: torch.Tensor,
) -> tuple[float, int]:
    """Compute BPB from per-token loss + output-position mask + output byte weights.

    Signature contract: ``byte_weights_output`` has shape ``(n_output,)`` and
    aligns with the True positions of ``mask`` in order (i.e.,
    ``loss_flat[mask]`` has the same length as ``byte_weights_output``).

    BPB = sum(loss[i] * byte_weight[i]) / (ln(2) * total_bytes)

    Raises:
        AxBenchBpbError: shape mismatch; non-finite loss at any in-mask
            position; total bytes == 0.
    """
    if loss_flat.shape != mask.shape:
        raise AxBenchBpbError(
            f"loss/mask shape mismatch: {loss_flat.shape} vs {mask.shape}"
        )
    n_mask_positions = int(mask.sum().item())
    if byte_weights_output.numel() != n_mask_positions:
        raise AxBenchBpbError(
            f"byte_weights_output length {byte_weights_output.numel()} "
            f"must match number of True positions in mask ({n_mask_positions})"
        )
    in_mask_losses = loss_flat[mask]
    if not torch.isfinite(in_mask_losses).all():
        bad = (~torch.isfinite(in_mask_losses)).sum().item()
        raise AxBenchBpbError(
            f"non-finite values in output-position loss: {bad} of {in_mask_losses.numel()}"
        )
    bytes_total = int(byte_weights_output.sum().item())
    # Key formula: nats = sum over output positions of (loss * byte_weight).
    # Use in_mask_losses (already extracted & checked) to avoid NaN
    # contamination from non-mask positions (NaN * 0 = NaN in IEEE 754).
    nats = (in_mask_losses.to(torch.float32) * byte_weights_output.to(torch.float32)).sum().item()
    if bytes_total == 0:
        raise AxBenchBpbError("zero bytes after masking; cannot compute BPB")
    bpb = nats / (math.log(2) * bytes_total)
    return bpb, bytes_total


@dataclass(frozen=True)
class TargetModelBundle:
    """Wraps a loaded target model + its tokenizer + config for scoring."""
    model: Any
    tokenizer: Any
    max_seq_len: int
    target_name: TargetModel | str
    cluster_contract: ClusterFeatureContract | None = None
    cluster_pre_forward_guard: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_name", TargetModel.coerce(self.target_name))

    def ensure_ready_for_forward(self) -> None:
        """Run the clustered checkpoint-byte pin immediately before first use."""
        if self.cluster_contract is None:
            return
        if self.cluster_pre_forward_guard is None:
            raise AxBenchBpbError(
                "clustered target bundle is missing its validated pre-forward "
                "checkpoint guard"
            )
        self.cluster_pre_forward_guard()


def _padded_idx_unique_per_row(
    seg_map: torch.Tensor, n_seg_max: int,
) -> None:
    """Recompute cross-attention's `padded_idx` and assert per-row uniqueness.

    Cross-attention's scatter step requires `padded_idx = seg_map * max_seg_size
    + pos_in_seg` to be unique within each row; otherwise overlapping writes
    silently corrupt real positions. This is the most subtle invariant in the
    batched microbatch path — `_pad_encoder_inputs` calls this guard after
    populating the padded `seg_map` with per-row generation_slot.

    Args:
        seg_map: (B, T) long tensor.
        n_seg_max: encoder's first non-batch dimension; needed for the
            collision-domain size check.

    Raises:
        AxBenchBpbError: the padded seg_map has at least one row where two
            (slot, pos_in_seg) pairs collide. Message names the offending
            row and the colliding positions.
    """
    if seg_map.dim() != 2:
        raise AxBenchBpbError(
            f"_padded_idx_unique_per_row: seg_map must be 2-D, got {seg_map.shape}"
        )
    B, T = seg_map.shape
    pos = torch.arange(T, device=seg_map.device).unsqueeze(0).expand(B, -1)
    is_seg_start = torch.zeros(B, T, dtype=torch.bool, device=seg_map.device)
    is_seg_start[:, 0] = True
    is_seg_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
    seg_start_pos = torch.where(
        is_seg_start, pos, torch.zeros_like(pos),
    ).cummax(dim=1).values
    pos_in_seg = pos - seg_start_pos
    # Domain: slot * (max_pos_in_seg + 1). Use seg_map.max+1 (not n_seg_max) for
    # the "stride" so we don't overcount; bug detection only needs uniqueness.
    max_seg_size_for_check = int(pos_in_seg.amax().item()) + 1
    padded_idx = seg_map * max_seg_size_for_check + pos_in_seg
    for b in range(B):
        row = padded_idx[b].tolist()
        if len(set(row)) != len(row):
            seen: dict[int, int] = {}
            for t, v in enumerate(row):
                if v in seen:
                    raise AxBenchBpbError(
                        f"padded_idx collision in row {b}: position "
                        f"{seen[v]} and position {t} both map to "
                        f"padded_idx={v}. seg_map row = {seg_map[b].tolist()}, "
                        f"n_seg_max={n_seg_max}. Likely cause: seg_map padded "
                        f"with a slot value that creates a second run of an "
                        f"already-used slot. Pad with per-row generation_slot."
                    )
                seen[v] = t


def _build_encdec_seg_map_and_encoder(
    *,
    tokenizer: Any,
    full_prefix: list[int],
    prefix_segments: tuple[ContextFeatureSegment, ...],
    feature_conditions: dict[int, int] | None,
    n_output: int,
    n_prompt_surviving: int,
    device: torch.device,
    allowed_ids: torch.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build (enc_feature_ids, enc_magnitudes, enc_mask, seg_map, generation_slot).

    Owns the full encoder feature-input + seg_map construction for the
    encdec teacher-forcing path used by both ``_pad_encoder_inputs`` (the
    batched microbatch kernel) and ``score_sample`` (the per-sample
    kernel). Returns tensors with batch dim 1 and last dim
    ``len(full_prefix) + n_output``.

    Branching:

    * **Shifted branch** — fires when ``n_prompt_surviving > 0`` AND
      ``prefix_segments`` is non-empty. Prepends an empty masked
      conditioning slot at encoder index 0; shifts every non-zero
      seg_map value up by 1; routes prompt-prefix tokens to slot 0
      (no encoder conditioning); routes inputs-prefix tokens to slots
      1..k; routes output tokens to slot k+1.

      The empty slot at index 0 has ``enc_mask`` all False, so the
      cross-attention scatter contributes zero to decoder positions
      reading from slot 0. Prompt-prefix tokens then receive no encoder
      conditioning — semantically correct because the prompt prefix
      isn't pretraining text with SAE features.

      The shifted seg_map satisfies both the monotonic-up-from-zero
      contract (``cross_attention.py:185-196``) and the
      ``_padded_idx_unique_per_row`` (G9) invariant. The alternative
      "route prompt-prefix to generation_slot" approach fails both
      contracts; the "route to slot 0 directly" approach silently
      double-attends to the first prefix segment.

    * **Existing branch** — fires otherwise. Calls
      ``build_multi_segment_seg_map`` directly on ``full_prefix`` with
      the original segment indices; encoder has ``len(prefix_segments) + 1``
      slots total.

    Args:
        tokenizer: Decoder tokenizer; passed through to
            ``build_multi_segment_seg_map``.
        full_prefix: Combined post-truncation prefix tokens
            (``prompt_prefix_tokens + inputs.prefix_tokens`` after
            ``truncate_prefix_left``).
        prefix_segments: Inputs-prefix-relative segments, already
            projected if truncation dropped inputs-prefix tokens.
        feature_conditions: ``intervention.feature_conditions`` or
            ``None``; typically ``{}`` for prompting interventions.
        n_output: Number of teacher-forced output tokens.
        n_prompt_surviving: Number of prompt-prefix tokens that survived
            ``truncate_prefix_left``. Equal to
            ``len(full_prefix) - n_inputs_surviving``.
        device: Target torch device for the returned tensors.
        allowed_ids: The checkpoint's compacted vocabulary ``V`` (original
            SAE feature ids, int64), or ``None`` for an uncompacted
            checkpoint. ``prefix_segments`` are context-cache features
            (raw, threshold-filtered SAE activations extracted from the
            prompt) that are NOT pre-selected against ``V`` the way a
            conditioning method's bag is, so they are filtered here, at
            the encoder-input boundary: ids outside ``V`` are dropped,
            with one ``logger.warning`` per call naming the total dropped
            count (aggregated across every segment in ``prefix_segments``,
            not one warning per segment). A segment that had feature ids
            but is fully emptied by the filter (every one of its ids was
            outside ``V``) raises ``AxBenchBpbError`` — a
            zero-conditionable-feature segment is an error, not a
            silently-empty one. A segment with no feature ids to begin
            with is unaffected (that is pre-existing, unrelated
            behaviour — it was never "emptied by the filter").
            ``feature_conditions`` (the generation slot) is NOT filtered
            here — the method's own bag is already restricted upstream.
            ``None`` is a no-op, byte-identical to before this parameter
            existed. See
            ``steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids``
            / ``restrict_to_vocab``.

    Returns:
        Tuple of:
            * ``enc_feature_ids``: ``(1, n_seg, F_max)`` int64.
            * ``enc_magnitudes``: ``(1, n_seg, F_max)`` int64.
            * ``enc_mask``: ``(1, n_seg, F_max)`` bool.
            * ``seg_map``: ``(1, len(full_prefix) + n_output)`` long.
            * ``generation_slot``: int — the slot index for output positions
              (used by ``_pad_encoder_inputs`` to pad the seg_map tail).

        ``n_seg`` is ``len(prefix_segments) + 1`` in the existing branch
        and ``len(prefix_segments) + 2`` in the shifted branch.

    Notes:
        Spec ref: ``docs/superpowers/specs/2026-05-11-prompting-encdec-encoder-conditioning-design.md``.
    """
    if not prefix_segments:
        raise AxBenchBpbError(
            "Encoder-decoder BPB scoring requires non-empty "
            "prefix_context_segments from the context cache. Got no prefix "
            "segments; rebuild/validate the AxBench-BPB context cache for "
            "this prefix instead of scoring with unconditioned prompt text."
        )

    if cluster_contract is not None:
        if allowed_ids is None:
            raise AxBenchBpbError(
                "clustered encoder packing requires the exact checkpoint "
                "FeatureVocab IDs"
            )
        contract_vocab = cluster_contract.checkpoint.feature_vocab
        actual_vocab = tuple(sorted(int(value) for value in allowed_ids.tolist()))
        if contract_vocab is None or actual_vocab != contract_vocab.original_ids:
            raise AxBenchBpbError(
                "clustered encoder packing checkpoint vocabulary differs from "
                "the validated feature contract"
            )
        for seg_idx, segment in enumerate(prefix_segments):
            ids = torch.tensor(segment.feature_ids, dtype=torch.int64)
            try:
                cluster_contract.consolidation.assert_representatives_only(
                    ids,
                    surface=f"encoder context segment {seg_idx}",
                )
                cluster_contract.feature_mask.assert_no_masked(ids)
            except Exception as exc:
                raise AxBenchBpbError(
                    "clustered encoder context must already be representative-only "
                    f"and mask-allowed before packing: {exc}"
                ) from exc
        generation_ids = torch.tensor(
            sorted((feature_conditions or {}).keys()),
            dtype=torch.int64,
        )
        try:
            cluster_contract.consolidation.assert_representatives_only(
                generation_ids,
                surface="encoder generation-slot conditioning",
            )
            cluster_contract.feature_mask.assert_no_masked(generation_ids)
        except Exception as exc:
            raise AxBenchBpbError(
                "clustered encoder generation bag violates the validated "
                f"representative contract: {exc}"
            ) from exc

    prefix_segment_conditions: list[dict[int, int]] = []
    total_dropped = 0
    for seg_idx, seg in enumerate(prefix_segments):
        pairs = [
            (int(fid), int(bkt))
            for fid, bkt in zip(seg.feature_ids, seg.buckets, strict=True)
        ]
        ids_tensor = torch.tensor(
            [fid for fid, _bkt in pairs], dtype=torch.int64,
        )
        kept, dropped = restrict_to_vocab(ids_tensor, allowed_ids)
        if dropped.numel() > 0:
            total_dropped += dropped.numel()
            if kept.numel() == 0:
                raise AxBenchBpbError(
                    f"Encoder-decoder BPB scoring: context-cache prefix "
                    f"segment {seg_idx} (char [{seg.char_start}, "
                    f"{seg.char_end})) has {len(pairs)} feature id(s), ALL "
                    f"outside the checkpoint's compacted vocabulary V "
                    f"(dropped ids: {sorted(dropped.tolist())}). A context "
                    f"segment with zero conditionable features is an "
                    f"error; rebuild/validate the AxBench-BPB context "
                    f"cache against this checkpoint's feature_vocab."
                )
        kept_ids = set(kept.tolist())
        prefix_segment_conditions.append(
            {fid: bkt for fid, bkt in pairs if fid in kept_ids}
        )
    if total_dropped > 0:
        if cluster_contract is not None:
            raise AxBenchBpbError(
                "clustered encoder packing attempted to filter context IDs at "
                "read time; the context cache is stale or corrupt"
            )
        logger.warning(
            f"Encoder-decoder BPB scoring: dropped {total_dropped} "
            f"context-cache feature id(s) outside the checkpoint's "
            f"compacted vocabulary V, across {len(prefix_segments)} "
            f"prefix segment(s)."
        )
    generation_slot_features: dict[int, int] = (
        dict(feature_conditions) if feature_conditions else {}
    )

    needs_empty_prompt_slot = n_prompt_surviving > 0 and bool(prefix_segments)

    if needs_empty_prompt_slot:
        # Empty masked slot at encoder index 0 reserves a no-encoder-context
        # routing target for prompt-prefix tokens. Cross-attention scatters
        # zero contribution for any decoder position whose seg_map points
        # to a fully-masked slot.
        record_counter(
            "bpb.prompting_encdec_empty_slot_emitted",
            1,
            n_prompt_surviving=n_prompt_surviving,
            n_segments=len(prefix_segments),
        )
        empty_slot: dict[int, int] = {}
        all_segment_conditions: list[dict[int, int] | None] = (
            [empty_slot] + prefix_segment_conditions + [generation_slot_features]
        )
        generation_slot = 1 + len(prefix_segments)

        # Inputs-prefix portion: assign segments [0..k-1] for inputs prefix
        # and ``len(prefix_segments)`` for output (the "original" generation
        # slot pre-shift).
        inputs_only_seg_map = build_multi_segment_seg_map(
            tokenizer=tokenizer,
            prefix_tokens=full_prefix[n_prompt_surviving:],
            prefix_segments=prefix_segments,
            n_output=n_output,
            generation_slot=len(prefix_segments),
        )
        # Shift every value up by 1 so segments live in [1..k] and the
        # generation slot lives at k+1. Slot 0 is now reserved.
        inputs_only_seg_map = inputs_only_seg_map + 1
        prompt_part = torch.zeros((1, n_prompt_surviving), dtype=torch.long)
        seg_map = torch.cat([prompt_part, inputs_only_seg_map], dim=1)
    else:
        all_segment_conditions: list[dict[int, int] | None] = (
            prefix_segment_conditions + [generation_slot_features]
        )
        generation_slot = len(prefix_segments)
        seg_map = build_multi_segment_seg_map(
            tokenizer=tokenizer,
            prefix_tokens=full_prefix,
            prefix_segments=prefix_segments,
            n_output=n_output,
            generation_slot=generation_slot,
        )

    enc_fids, enc_mags, enc_mask = build_segmented_feature_encoder_inputs(
        all_segment_conditions, device,
    )
    seg_map = seg_map.to(device)

    return enc_fids, enc_mags, enc_mask, seg_map, generation_slot


def assert_enc_ids_in_vocab(
    enc_feature_ids: torch.Tensor,
    enc_mask: torch.Tensor,
    allowed_ids: torch.Tensor | None,
) -> None:
    """Eager pre-forward backstop: every REAL id in ``enc_feature_ids`` must be in ``V``.

    Called immediately before every encoder-decoder ``model(enc_feature_ids=...)``
    forward in this package (``score_sample``, ``_score_microbatch``, and
    ``steering.eval.axbench_bpb_adv.surrogate._teacher_forced_model_forward``).
    The feature-DiffMean/general-val bag and the context-cache features are
    already restricted to the checkpoint's compacted vocabulary ``V`` well
    before they reach here (see
    ``steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`` /
    ``restrict_to_vocab``, and this module's own
    ``_build_encdec_seg_map_and_encoder`` filter), so in a correct pipeline
    this never fires. It exists to catch a missed — or future — restriction
    site LOUDLY, in Python, at the exact call site that is about to feed the
    model a bad id, rather than relying on ``FeatureEncoder._to_compact``'s
    ``torch._assert_async`` deep inside the forward as the only backstop:
    that guard is deliberately sync-free (zero graph breaks under
    ``torch.compile``), so a violation there can surface far from its actual
    cause. See ``steering.conditional.encoder_decoder.encoder.py::_to_compact``.

    Only ``enc_mask``-True ("real") slots are checked. Padding slots
    (``enc_mask`` False) are ignored unconditionally: every producer in this
    package pads ``enc_feature_ids`` with literal ``0`` — a real SAE id a
    whitelist may legitimately exclude — so requiring padding to also sit
    inside ``V`` would make an ordinary padded microbatch spuriously fail.

    ``allowed_ids=None`` (legacy, uncompacted checkpoint) is a true no-op:
    no tensor op runs, so the check is free on the byte-identical legacy
    path. Device-safety (``enc_feature_ids``/``allowed_ids`` need not share a
    device) is inherited from :func:`restrict_to_vocab`, which this reuses
    rather than reimplementing.

    Raises:
        AxBenchBpbError: at least one ``enc_mask``-True slot holds an id not
            in ``allowed_ids``. The message names the offending id(s).
    """
    if allowed_ids is None:
        return
    real_ids = enc_feature_ids[enc_mask]
    _kept, dropped = restrict_to_vocab(real_ids, allowed_ids)
    if dropped.numel() == 0:
        return
    raise AxBenchBpbError(
        f"enc_feature_ids: real (non-padding) slot(s) hold id(s) outside "
        f"the checkpoint's compacted vocabulary V (offending ids: "
        f"{sorted(dropped.tolist())}). This is a correctness backstop — "
        f"every enc_feature_ids producer is expected to already restrict "
        f"candidate ids to V upstream (see "
        f"steering.eval.axbench_bpb.vocab_restriction); reaching this "
        f"assert means some site was missed."
    )


def assert_enc_ids_match_cluster_contract(
    enc_feature_ids: torch.Tensor,
    enc_mask: torch.Tensor,
    cluster_contract: ClusterFeatureContract,
) -> None:
    """Assert representative-only clustered IDs immediately before a forward."""

    real_ids = enc_feature_ids[enc_mask].detach().to(device="cpu", dtype=torch.int64)
    try:
        cluster_contract.consolidation.assert_representatives_only(
            real_ids,
            surface="encoder pre-forward packed IDs",
        )
        cluster_contract.feature_mask.assert_no_masked(real_ids)
    except Exception as exc:
        raise AxBenchBpbError(
            "encoder pre-forward packed IDs violate the validated cluster "
            f"contract: {exc}"
        ) from exc


def _pad_decoder_inputs(
    samples: list[ScoringInputs],
    intervention: Intervention,
    target_bundle: TargetModelBundle,
    device: torch.device,
) -> _DecoderPaddedInputs:
    """Build the padded (B, T_max-1) decoder tensors common to both targets.

    Per-sample steps mirror score_sample steps 1-4:
      1. Fold intervention.prompt_prefix_tokens into prefix (if non-None).
      2. Left-truncate via truncate_prefix_left to fit max_seq_len.
      3. Build full_tokens = full_prefix + output_tokens; shift-by-1.
      4. Build per-sample output-only mask via build_output_only_mask.

    Padding (across the microbatch):
      - dec_input  pad value: 0           (any in-vocab id; loss-masked)
      - dec_target pad value: -1          (model.cross_entropy ignore_index)
      - loss_mask  pad value: False       (accumulate_bpb extracts True positions)

    Single-shot, single-threaded use only. Holds no cross-call state.
    """
    if not samples:
        raise AxBenchBpbError("_pad_decoder_inputs: samples must be non-empty")

    full_prefixes: list[list[int]] = []
    n_full_prefix_list: list[int] = []
    n_output_list: list[int] = []
    truncated_list: list[bool] = []
    masks: list[torch.Tensor] = []

    for s in samples:
        full_prefix = list(s.prefix_tokens)
        if intervention.prompt_prefix_tokens:
            full_prefix = list(intervention.prompt_prefix_tokens) + full_prefix
        full_prefix, truncated_after_prompt = truncate_prefix_left(
            prefix=full_prefix, n_output=s.n_output,
            max_seq_len=target_bundle.max_seq_len,
        )
        try:
            mask = build_output_only_mask(
                n_full_prefix=PositiveInt.parse(len(full_prefix)),
                n_output=PositiveInt.parse(s.n_output),
            )
        except TypeError as exc:
            raise AxBenchBpbError(
                f"{exc} (sample_id={s.sample_id!r})"
            ) from exc

        full_prefixes.append(full_prefix)
        n_full_prefix_list.append(len(full_prefix))
        n_output_list.append(s.n_output)
        truncated_list.append(s.truncated or truncated_after_prompt)
        masks.append(mask)

    T_max = max(n_full_prefix_list[i] + n_output_list[i] for i in range(len(samples)))
    B = len(samples)

    dec_input = torch.zeros(B, T_max - 1, dtype=torch.long, device=device)
    dec_target = torch.full(
        (B, T_max - 1), -1, dtype=torch.long, device=device,
    )
    loss_mask = torch.zeros(B, T_max - 1, dtype=torch.bool, device=device)

    for i, s in enumerate(samples):
        all_tokens = full_prefixes[i] + list(s.output_tokens)
        T_i = len(all_tokens)  # == n_full_prefix_list[i] + n_output_list[i]
        dec_input[i, : T_i - 1] = torch.tensor(
            all_tokens[:-1], dtype=torch.long, device=device,
        )
        dec_target[i, : T_i - 1] = torch.tensor(
            all_tokens[1:], dtype=torch.long, device=device,
        )
        loss_mask[i, : T_i - 1] = masks[i].to(device)

    return _DecoderPaddedInputs(
        dec_input=dec_input,
        dec_target=dec_target,
        loss_mask=loss_mask,
        n_full_prefix=n_full_prefix_list,
        n_output=n_output_list,
        truncated=truncated_list,
        full_prefixes=full_prefixes,
        T_max=T_max,
    )


def _pad_encoder_inputs(
    samples: list[ScoringInputs],
    full_prefixes: list[list[int]],
    intervention: Intervention,
    target_bundle: TargetModelBundle,
    T_max: int,
    device: torch.device,
) -> _EncoderPaddedInputs:
    """Build padded (B, n_seg_max, F_max) encoder tensors and (B, T_max-1) seg_map.

    Per-sample steps mirror score_sample step 5 (with the multi-segment fix):
      - segment_conditions = [{fid: bucket} for seg in PCS] + [feature_conditions or {}]
      - enc_*_i = build_segmented_feature_encoder_inputs(segment_conditions, device)
      - seg_map_i = build_multi_segment_seg_map(...) [:, :-1]

    Padding (across the microbatch):
      - enc_*   pad value: zeros for ids/mags, False for mask, on (B, n_seg_max, F_max)
      - seg_map pad value: per-row generation_slot_i in the [T_i-1, T_max-1) tail
                           (avoids cross-attention scatter collisions)

    Asserts `_padded_idx_unique_per_row` AFTER padding (G9).
    Asserts `seg_map[i].max() < n_seg_max` per row (G10).
    """
    if target_bundle.target_name != TargetModel.ENCDEC:
        raise AxBenchBpbError(
            f"_pad_encoder_inputs called with target_name={target_bundle.target_name!r}; "
            f"only 'encdec' is supported"
        )
    if len(samples) != len(full_prefixes):
        raise AxBenchBpbError(
            f"_pad_encoder_inputs: samples ({len(samples)}) and full_prefixes "
            f"({len(full_prefixes)}) length mismatch"
        )

    B = len(samples)
    per_sample_enc: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    per_sample_seg_map: list[torch.Tensor] = []
    per_sample_gen_slot: list[int] = []

    # Compacted checkpoint's allowed vocabulary V (None if uncompacted).
    # Computed once — target_bundle.model is constant across the loop.
    allowed_ids = feature_vocab_allowed_ids(target_bundle.model)

    n_prompt_prefix_tokens = (
        len(intervention.prompt_prefix_tokens)
        if intervention.prompt_prefix_tokens else 0
    )

    for i, s in enumerate(samples):
        # Q11 (slice 17 audit 2026-05-05): project segments onto the
        # post-truncation coordinate space if truncation dropped tokens of
        # the original ``s.prefix_tokens``. ``full_prefixes[i]`` is the
        # post-truncation prefix; total combined-prefix tokens before
        # truncation == n_prompt_prefix_tokens + len(s.prefix_tokens).
        truncated_prefix_segments = s.prefix_context_segments
        n_inputs_prefix = len(s.prefix_tokens)
        n_remaining_in_full = len(full_prefixes[i])
        tokens_dropped_total = (
            n_prompt_prefix_tokens + n_inputs_prefix - n_remaining_in_full
        )
        tokens_of_inputs_dropped = max(
            0, tokens_dropped_total - n_prompt_prefix_tokens
        )
        n_inputs_surviving = n_inputs_prefix - tokens_of_inputs_dropped
        n_prompt_surviving = n_remaining_in_full - n_inputs_surviving
        if s.prefix_context_segments and tokens_of_inputs_dropped > 0:
            truncated_inputs_prefix = list(
                s.prefix_tokens
            )[tokens_of_inputs_dropped:]
            truncated_prefix_segments = project_segments_to_truncated_prefix(
                s.prefix_context_segments,
                original_prefix_tokens=list(s.prefix_tokens),
                truncated_prefix_tokens=truncated_inputs_prefix,
                tokenizer=target_bundle.tokenizer,
            )

        # Build encoder feature tensors + seg_map via the shared helper.
        # When prompt-prefix tokens survive AND segments are non-empty,
        # the helper prepends an empty masked encoder slot at index 0 and
        # shifts non-zero seg_map values up by 1 so that prompt-prefix
        # tokens route to the empty slot (no encoder conditioning) and
        # inputs-prefix tokens still receive their proper segment
        # conditioning. See
        # docs/superpowers/specs/2026-05-11-prompting-encdec-encoder-conditioning-design.md.
        enc_fids_i, enc_mags_i, enc_mask_i, seg_map_i, generation_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=target_bundle.tokenizer,
                full_prefix=full_prefixes[i],
                prefix_segments=truncated_prefix_segments,
                feature_conditions=intervention.feature_conditions,
                n_output=s.n_output,
                n_prompt_surviving=n_prompt_surviving,
                device=device,
                allowed_ids=allowed_ids,
                cluster_contract=target_bundle.cluster_contract,
            )
        )
        per_sample_enc.append((enc_fids_i, enc_mags_i, enc_mask_i))
        per_sample_gen_slot.append(generation_slot)
        seg_map_i = seg_map_i[:, :-1]  # drop final position (no decoder target)
        per_sample_seg_map.append(seg_map_i)

    n_seg_max = max(t[0].shape[1] for t in per_sample_enc)
    F_max = max(t[0].shape[2] for t in per_sample_enc)

    enc_fids = torch.zeros(B, n_seg_max, F_max, dtype=torch.int64, device=device)
    enc_mags = torch.zeros(B, n_seg_max, F_max, dtype=torch.int64, device=device)
    enc_mask = torch.zeros(B, n_seg_max, F_max, dtype=torch.bool, device=device)
    for i, (fids_i, mags_i, mask_i) in enumerate(per_sample_enc):
        n_seg_i = fids_i.shape[1]
        F_i = fids_i.shape[2]
        enc_fids[i, :n_seg_i, :F_i] = fids_i[0]
        enc_mags[i, :n_seg_i, :F_i] = mags_i[0]
        enc_mask[i, :n_seg_i, :F_i] = mask_i[0]

    seg_map = torch.zeros(B, T_max - 1, dtype=torch.long, device=device)
    for i in range(B):
        sm_i = per_sample_seg_map[i]  # (1, T_i - 1)
        T_i_minus_1 = sm_i.shape[1]
        seg_map[i, :T_i_minus_1] = sm_i[0]
        # Pad tail with per-row generation_slot (extends the last real run; no
        # pos_in_seg reset; no scatter collision).
        seg_map[i, T_i_minus_1:] = per_sample_gen_slot[i]

        # G10: assert in-range.
        if int(seg_map[i].max().item()) >= n_seg_max:
            raise AxBenchBpbError(
                f"_pad_encoder_inputs row {i}: seg_map.max()="
                f"{int(seg_map[i].max().item())} >= n_seg_max={n_seg_max} "
                f"(sample_id={samples[i].sample_id!r}, "
                f"len(prefix_context_segments)={len(samples[i].prefix_context_segments)})"
            )

    # G9: assert padded_idx unique per row.
    _padded_idx_unique_per_row(seg_map, n_seg_max)

    return _EncoderPaddedInputs(
        enc_feature_ids=enc_fids,
        enc_magnitudes=enc_mags,
        enc_mask=enc_mask,
        seg_map=seg_map,
        allowed_ids=allowed_ids,
    )


def _extract_per_sample_bpb(
    loss_b: torch.Tensor,
    padded_decoder: _DecoderPaddedInputs,
    samples: list[ScoringInputs],
    target_name: TargetModel | str,
    coefficient_str: str,
    method_name: str,
) -> list[BpbScoreRow]:
    """Per-sample BPB extraction via accumulate_bpb on each row's slice.

    Mirrors score_sample step 8 per row, with sample_id-enriched error wrapping.
    """
    if loss_b.dim() != 2:
        raise AxBenchBpbError(
            f"_extract_per_sample_bpb: loss_b must be 2-D, got {loss_b.shape}"
        )
    if loss_b.shape[0] != len(samples):
        raise AxBenchBpbError(
            f"_extract_per_sample_bpb: loss_b batch dim {loss_b.shape[0]} "
            f"!= len(samples) {len(samples)}"
        )
    if loss_b.shape[1] != padded_decoder.loss_mask.shape[1]:
        raise AxBenchBpbError(
            f"_extract_per_sample_bpb: loss_b T-dim {loss_b.shape[1]} "
            f"!= loss_mask T-dim {padded_decoder.loss_mask.shape[1]}"
        )

    target = TargetModel.coerce(target_name)
    rows: list[BpbScoreRow] = []
    device = loss_b.device
    for i, s in enumerate(samples):
        loss_i_flat = loss_b[i]
        mask_i = padded_decoder.loss_mask[i]
        byte_weights_i = s.output_byte_weights.to(device)
        try:
            bpb, n_bytes = accumulate_bpb(loss_i_flat, mask_i, byte_weights_i)
        except AxBenchBpbError as exc:
            raise AxBenchBpbError(
                f"{exc} (sample_id={s.sample_id!r}, method={method_name!r}, "
                f"coefficient={coefficient_str!r})"
            ) from exc
        rows.append(BpbScoreRow(
            method_name=method_name,
            concept_id=s.concept_id,
            mode=s.mode.value,
            content_class=s.content_class,
            target=target,
            coefficient_str=coefficient_str,
            sample_id=s.sample_id,
            bpb_intervened=bpb,
            n_output_tokens=s.n_output,
            n_output_bytes=n_bytes,
            truncated=padded_decoder.truncated[i],
        ))
    return rows


@torch.no_grad()
def score_sample(
    *,
    inputs: ScoringInputs,
    target_bundle: TargetModelBundle,
    intervention: Intervention,
    device: torch.device,
    coefficient_str: str,
    method_name: str,
    lora: LoraParams | None = None,
) -> BpbScoreRow:
    """Teacher-force one sample through the target model under the intervention.

    Steps (spec §8):
      1. Fold in prompting prompt_prefix_tokens if present.
      2. Left-truncate prefix if prefix+output exceeds max_seq_len.
      3. Shift-by-1 → dec_input, dec_target.
      4. Build output-only loss mask.
      5. For encdec targets: build encoder feature inputs + seg_map.
      6. Apply intervention (context manager; hook/adapter installed).
      7. Forward the model with loss_reduction="none" → (1, T-1) float32.
      8. Accumulate BPB from in-mask losses and byte weights.
      9. Emit BpbScoreRow.

    Args:
        inputs: Preprocessed sample (prefix tokens, output tokens, byte weights,
            context segments, ids).
        target_bundle: Loaded model + tokenizer + config.
        intervention: Intervention object (from build_intervention).
        device: Target device for tensor construction.
        coefficient_str: String representation of coefficient for leaderboard keying
            (e.g. "1.0", "None" for LoRA/prompting).
        method_name: Name for the emitted BpbScoreRow (from the method registry).
        lora: LoRA attach params bundle (cfg + layer); only needed when
            intervention.kind == "lora", ``None`` for the other kinds.

    Returns:
        BpbScoreRow with the measured BPB.

    Raises:
        AxBenchBpbError: non-finite loss; zero-byte output; shape mismatch.
    """
    # 1. Prompting fold-in
    full_prefix = list(inputs.prefix_tokens)
    if intervention.prompt_prefix_tokens:
        full_prefix = list(intervention.prompt_prefix_tokens) + full_prefix

    # 2. Left-truncate to fit context
    pre_truncation_full_prefix = full_prefix
    full_prefix, truncated_after_prompt = truncate_prefix_left(
        prefix=full_prefix, n_output=inputs.n_output,
        max_seq_len=target_bundle.max_seq_len,
    )
    truncated = inputs.truncated or truncated_after_prompt
    n_full_prefix = len(full_prefix)
    # Q11 (slice 17 audit 2026-05-05): if truncation actually dropped tokens
    # AND the prefix carries char-offset-based segments, project segments
    # into the truncated coordinate space. Only the encdec branch uses
    # ``inputs.prefix_context_segments``; project here so the projection
    # is shared with the prompt-prefix-token path (whose char offsets the
    # segments must NOT see — see note below).
    #
    # Note on prompt-prefix-token semantics: ``intervention.prompt_prefix_tokens``
    # is unrelated to ``inputs.prefix_context_segments``. The stored segments
    # were extracted against ``inputs.prefix_text`` (Stage 1) and their
    # ``char_start`` / ``char_end`` are offsets into that text, NOT into a
    # synthesized prompt-prefix string. When prompt tokens are folded in on
    # the left, segments either (a) survive unchanged if no truncation
    # actually removed prefix chars, or (b) need their char offsets shifted
    # by the difference between the original-prefix-tokens length and the
    # post-truncation tokens that ARE drawn from the original prefix.
    #
    # Concretely: if ``intervention.prompt_prefix_tokens`` has length P,
    # and we truncate to keep only K rightmost tokens of the (P + N_orig)
    # combined prefix, then the dropped tokens come from the leftmost
    # (P + N_orig - K). After truncation, the ``segments`` list (built
    # against ``inputs.prefix_tokens``, not the prompt) needs its char
    # offsets shifted by:
    #   dropped_chars_within_inputs_prefix
    # i.e. how many chars of ``inputs.prefix_text`` were dropped. We
    # compute this by re-running ``token_to_char_offsets`` against
    # ``inputs.prefix_tokens`` alone (NOT the combined prefix), with
    # n_dropped_within_inputs = max(0, len(inputs.prefix_tokens) - len(remaining_inputs_tokens)).
    # Q11 (slice 17 audit 2026-05-05) + audit-fixup 2026-05-11: compute
    # counters for the prompt-prefix + segments routing decision and the
    # post-truncation projection.
    truncated_prefix_segments = inputs.prefix_context_segments
    n_prompt_surviving = 0
    if target_bundle.target_name == TargetModel.ENCDEC:
        n_prompt = (
            len(intervention.prompt_prefix_tokens)
            if intervention.prompt_prefix_tokens else 0
        )
        n_inputs_prefix = len(inputs.prefix_tokens)
        n_remaining_in_full = len(full_prefix)
        tokens_dropped_total = (n_prompt + n_inputs_prefix) - n_remaining_in_full
        tokens_of_inputs_dropped = max(0, tokens_dropped_total - n_prompt)
        n_inputs_surviving = n_inputs_prefix - tokens_of_inputs_dropped
        n_prompt_surviving = n_remaining_in_full - n_inputs_surviving
        if (
            inputs.prefix_context_segments
            and len(full_prefix) < len(pre_truncation_full_prefix)
            and tokens_of_inputs_dropped > 0
        ):
            # Build a synthetic "post-truncation inputs prefix" that retains
            # only the last (n_inputs_prefix - tokens_of_inputs_dropped)
            # tokens of inputs.prefix_tokens. This is the coordinate space
            # the surviving segments must live in.
            truncated_inputs_prefix = list(
                inputs.prefix_tokens
            )[tokens_of_inputs_dropped:]
            truncated_prefix_segments = project_segments_to_truncated_prefix(
                inputs.prefix_context_segments,
                original_prefix_tokens=list(inputs.prefix_tokens),
                truncated_prefix_tokens=truncated_inputs_prefix,
                tokenizer=target_bundle.tokenizer,
            )
        # else: all dropped tokens were prompt-prefix-tokens (or no
        # truncation occurred), so segments stay unchanged.

    # 3. Shift-by-1
    all_tokens = full_prefix + list(inputs.output_tokens)
    dec_input = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=device)
    dec_target = torch.tensor([all_tokens[1:]], dtype=torch.long, device=device)

    # 4. Output-only loss mask
    mask = build_output_only_mask(
        n_full_prefix=PositiveInt.parse(n_full_prefix),
        n_output=PositiveInt.parse(inputs.n_output),
    ).to(device)

    # 5. Encoder inputs + seg_map for encdec targets only
    enc_fids = enc_mags = enc_mask = seg_map = allowed_ids = None
    if target_bundle.target_name == TargetModel.ENCDEC:
        # Build encoder feature tensors + seg_map via the shared helper.
        # When prompt-prefix tokens survive AND segments are non-empty,
        # the helper prepends an empty masked encoder slot at index 0 and
        # shifts non-zero seg_map values up by 1 so that prompt-prefix
        # tokens route to the empty slot (no encoder conditioning) and
        # inputs-prefix tokens still receive their proper segment
        # conditioning. See
        # docs/superpowers/specs/2026-05-11-prompting-encdec-encoder-conditioning-design.md.
        # `allowed_ids` is derived ONCE and reused below for the eager
        # pre-forward vocabulary assert (avoids a second
        # feature_vocab_ids read/derivation for the same call).
        allowed_ids = feature_vocab_allowed_ids(target_bundle.model)
        enc_fids, enc_mags, enc_mask, seg_map, _generation_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=target_bundle.tokenizer,
                full_prefix=full_prefix,
                prefix_segments=truncated_prefix_segments,
                feature_conditions=intervention.feature_conditions,
                n_output=inputs.n_output,
                n_prompt_surviving=n_prompt_surviving,
                device=device,
                allowed_ids=allowed_ids,
                cluster_contract=target_bundle.cluster_contract,
            )
        )
        seg_map = seg_map[:, :-1]

    # 6-7. Apply intervention + forward
    with apply_intervention(
        target_bundle.model, intervention,
        lora=lora,
    ):
        if target_bundle.target_name == TargetModel.ENCDEC:
            assert enc_fids is not None and enc_mask is not None
            # Eager backstop (issue #695 PR-1 Task 4): every REAL
            # (enc_mask-True) id must be in the checkpoint's compacted
            # vocabulary V, checked immediately before this forward — see
            # assert_enc_ids_in_vocab's docstring for why this exists
            # alongside FeatureEncoder._to_compact's own (async) OOV assert.
            assert_enc_ids_in_vocab(enc_fids, enc_mask, allowed_ids)
            if target_bundle.cluster_contract is not None:
                assert_enc_ids_match_cluster_contract(
                    enc_fids,
                    enc_mask,
                    target_bundle.cluster_contract,
                )
            target_bundle.ensure_ready_for_forward()
            loss = target_bundle.model(
                dec_input, dec_target,
                enc_feature_ids=enc_fids, enc_magnitudes=enc_mags,
                enc_mask=enc_mask, seg_map=seg_map,
                loss_reduction="none",
            )
        else:
            target_bundle.ensure_ready_for_forward()
            loss = target_bundle.model(
                dec_input, targets=dec_target, loss_reduction="none",
            )
    loss_flat = loss.view(-1)

    # 8. Accumulate BPB
    byte_weights_output = inputs.output_byte_weights.to(device)
    try:
        bpb, n_bytes = accumulate_bpb(loss_flat, mask, byte_weights_output)
    except AxBenchBpbError as exc:
        raise AxBenchBpbError(
            f"{exc} (sample_id={inputs.sample_id!r}, method={method_name!r}, "
            f"coefficient={coefficient_str!r})"
        ) from exc

    # 9. Emit row
    return BpbScoreRow(
        method_name=method_name,
        concept_id=inputs.concept_id,
        mode=inputs.mode.value,
        content_class=inputs.content_class,
        target=target_bundle.target_name,
        coefficient_str=coefficient_str,
        sample_id=inputs.sample_id,
        bpb_intervened=bpb,
        n_output_tokens=inputs.n_output,
        n_output_bytes=n_bytes,
        truncated=truncated,
    )


def _take_intervention_fingerprint(model: Any) -> tuple:
    """O(n_layers) snapshot of intervention-relevant model state.

    Captures (per transformer block):
      - len(block._forward_hooks)         — detects register_residual_hook leaks
      - len(block._forward_pre_hooks)     — defensive for future intervention kinds
      - tuple of attn submodule type names — detects LoRA wrap (LoRAAdapter
        replaces nn.Linear instances inside the attn module)

    Constant cost per microbatch — typically n_layers ∈ {12..32}; suitable
    for production use on every score_triple call.
    """
    return tuple(
        (
            i,
            len(block._forward_hooks),
            len(block._forward_pre_hooks),
            tuple(
                type(m).__name__
                for _, m in block.attn.named_modules()
                if m is not block.attn
            ),
        )
        for i, block in enumerate(model.transformer.h)
    )


@torch.no_grad()
def score_triple(
    *,
    samples: list[ScoringInputs],
    target_bundle: TargetModelBundle,
    intervention: Intervention,
    device: torch.device,
    coefficient_str: str,
    method_name: str,
    device_batch_size: int,
    lora: LoraParams | None = None,
) -> list[BpbScoreRow]:
    """Batched teacher-forced scoring of N samples under one intervention.

    Microbatches `samples` at `device_batch_size` and runs each microbatch
    through `_score_microbatch`. Wraps the entire call in a fingerprint
    pre/post check (G16) that catches any intervention which fails to
    tear down its model state.

    Raises:
        AxBenchBpbError: device_batch_size < 1, samples is empty, or the
            intervention left state behind (G16).
    """
    if device_batch_size < 1:
        raise AxBenchBpbError(
            f"device_batch_size must be >= 1, got {device_batch_size}"
        )
    if not samples:
        raise AxBenchBpbError(
            "score_triple: samples must be non-empty (caller filtered to zero work units?)"
        )

    fingerprint_before = _take_intervention_fingerprint(target_bundle.model)
    rows: list[BpbScoreRow] = []
    for start in range(0, len(samples), device_batch_size):
        chunk = samples[start : start + device_batch_size]
        rows.extend(_score_microbatch(
            samples=chunk, target_bundle=target_bundle, intervention=intervention,
            device=device, coefficient_str=coefficient_str, method_name=method_name,
            lora=lora,
        ))
    fingerprint_after = _take_intervention_fingerprint(target_bundle.model)

    if fingerprint_before != fingerprint_after:
        raise AxBenchBpbError(
            f"intervention {intervention.kind!r} left state behind. "
            f"Before: {fingerprint_before}. After: {fingerprint_after}. "
            f"This indicates apply_intervention's teardown is broken."
        )
    return rows


def _score_microbatch(
    *,
    samples: list[ScoringInputs],
    target_bundle: TargetModelBundle,
    intervention: Intervention,
    device: torch.device,
    coefficient_str: str,
    method_name: str,
    lora: LoraParams | None = None,
) -> list[BpbScoreRow]:
    """Score one micro-batch under a single intervention install.

    Single-shot, single-threaded. Pads decoder + encoder inputs to the
    microbatch's max shape, runs ONE forward, extracts per-sample BPB.

    All samples in the microbatch share `intervention` by signature; this is
    enforced by `score_triple`'s outer loop and the orchestrator
    (`run_bpb_score_stage`) which builds one Intervention per
    (method, concept, coefficient) triple.
    """
    if not samples:
        raise AxBenchBpbError("_score_microbatch: samples must be non-empty")

    padded_decoder = _pad_decoder_inputs(samples, intervention, target_bundle, device)
    B = len(samples)
    T_max = padded_decoder.T_max

    if target_bundle.target_name == TargetModel.ENCDEC:
        padded_encoder = _pad_encoder_inputs(
            samples, padded_decoder.full_prefixes, intervention,
            target_bundle, T_max, device,
        )
    else:
        padded_encoder = None

    with apply_intervention(
        target_bundle.model, intervention,
        lora=lora,
    ):
        if target_bundle.target_name == TargetModel.ENCDEC:
            assert padded_encoder is not None
            # Eager backstop (issue #695 PR-1 Task 4): reuses the
            # allowed_ids already derived inside _pad_encoder_inputs
            # (no second feature_vocab_ids read) — see
            # assert_enc_ids_in_vocab's docstring.
            assert_enc_ids_in_vocab(
                padded_encoder.enc_feature_ids,
                padded_encoder.enc_mask,
                padded_encoder.allowed_ids,
            )
            if target_bundle.cluster_contract is not None:
                assert_enc_ids_match_cluster_contract(
                    padded_encoder.enc_feature_ids,
                    padded_encoder.enc_mask,
                    target_bundle.cluster_contract,
                )
            target_bundle.ensure_ready_for_forward()
            loss = target_bundle.model(
                padded_decoder.dec_input, padded_decoder.dec_target,
                enc_feature_ids=padded_encoder.enc_feature_ids,
                enc_magnitudes=padded_encoder.enc_magnitudes,
                enc_mask=padded_encoder.enc_mask,
                seg_map=padded_encoder.seg_map,
                loss_reduction="none",
            )
        else:
            target_bundle.ensure_ready_for_forward()
            loss = target_bundle.model(
                padded_decoder.dec_input,
                targets=padded_decoder.dec_target,
                loss_reduction="none",
            )

    if not isinstance(loss, torch.Tensor):
        raise AxBenchBpbError(
            f"_score_microbatch: model returned {type(loss).__name__}; "
            f"expected torch.Tensor. Likely cause: "
            f"model.feature_pred_config is set and the model returned a "
            f"ForwardResult NamedTuple. axbench-bpb does not pass fp_targets, "
            f"so this should not happen — check model construction."
        )
    expected_shape = (B * (T_max - 1),)
    if loss.shape != expected_shape:
        raise AxBenchBpbError(
            f"_score_microbatch: loss.shape={tuple(loss.shape)} != "
            f"expected {expected_shape} (B={B}, T_max={T_max}). "
            f"Likely model API drift on loss_reduction='none' return shape."
        )

    loss_b = loss.view(B, T_max - 1)
    return _extract_per_sample_bpb(
        loss_b, padded_decoder, samples,
        target_bundle.target_name, coefficient_str, method_name,
    )


__all__ = [
    "TargetModelBundle",
    "accumulate_bpb",
    "assert_enc_ids_in_vocab",
    "assert_enc_ids_match_cluster_contract",
    "build_output_only_mask",
    "score_sample",
    "score_triple",
]
