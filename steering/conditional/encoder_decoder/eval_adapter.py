"""
Evaluation adapters for EncoderDecoderGPT models.

:class:`TargetAwareEncoderDecoderEvalWrapper` is the primary eval adapter.  It
routes decoder tokens through real Gemma-extracted segment character boundaries
converted into nanochat token space.  Prompt/context tokens attend to their
matching extracted context segment; generated/scored target tokens attend to
explicit target segment slots appended after the context slots.  It also
supports next-segment target timing, where a decoder segment attends to the
following segment's feature signal.

Helper function :func:`build_segmented_feature_encoder_inputs` builds padded
encoder inputs for multiple eval segments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from phantom.sized import NonEmptyStr

from nanochat.tokenizer import SPECIAL_TOKENS
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.full_sequence_routing import (
    context_token_assignments,
)
from steering.conditional.encoder_decoder.model import fp_head_unavailable_error
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    repeat_packed_encoder_output,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondRuntime,
    SelfCondState,
    _require_nonempty_self_cond_state,
    final_real_prompt_source_mask,
    self_conditioned_decode_step,
)
from steering.conditional.sampling import SamplingBundle
from steering.eval.target_conditioning import (
    ContextFeatureSegment as _ContextFeatureSegment,
)
from steering.eval.target_conditioning import (
    ContextRoutingMode,
    MultiTargetSegments,
    PreparedTurnContext,
    SelfConditionedTarget,
    SingleTargetSegment,
    TargetConditioning,
    TargetConditioningMode,
    context_cache_generation_instructions,
    token_context_key,
    validate_feature_dict,
)
from steering.eval.turn_context_contract import TurnContextContract, validate_turn_context_contract
from steering.eval.turn_context_routing import prepare_turn_context
from steering.profiling.markers import timed_region
from steering.segmentation import (
    SegmentMode,
    align_segments_to_decoded_prefix,
    compute_tokenizer_checksum,
    token_to_char_offsets,
)
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    legacy_turn_spans,
)

FeatureConditions = dict[int, int] | None
_CONTEXT_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.encdec_eval_context"
)
_BOS_TOKEN_NAME = "<bos>"
# The chat control tokens a turn-mode completion must stop on are exactly the
# vendored renderer's own special tokens.  BOS is resolved separately through
# ``tokenizer.get_bos_token_id()``, which is the only supported spelling for
# tokenizers whose BOS is not literally named ``<bos>``.
_TURN_STRUCTURAL_TOKEN_NAMES = tuple(name for name in SPECIAL_TOKENS if name != _BOS_TOKEN_NAME)


def _positive_count_policy(config: SelfCondConfig) -> object:
    return config.selection.selection


def _cluster_descriptor(config: SelfCondConfig) -> object | None:
    consolidation = config.cluster_consolidation
    return None if consolidation is None else consolidation.resolved.descriptor


def _percentile_table_id(config: SelfCondConfig) -> int | None:
    selection = config.selection
    return id(selection.percentile_table) if isinstance(selection, SamplingBundle) else None


def _selection_rng_id(config: SelfCondConfig) -> int | None:
    selection = config.selection
    return id(selection.rng) if isinstance(selection, SamplingBundle) else None


def _precomputed_thresholds_id(config: SelfCondConfig) -> int | None:
    selection = config.selection
    if isinstance(selection, SamplingBundle):
        return None
    thresholds = selection.precomputed_thresholds
    return None if thresholds is None else id(thresholds)


def _fp_logit_bias_id(config: SelfCondConfig) -> int | None:
    bias = config.fp_logit_bias
    return None if bias is None else id(bias)


def _self_cond_compat_key(
    provider: SelfConditionedFeatureProvider,
    config: SelfCondConfig,
) -> tuple[tuple[str, object], ...]:
    """Tensor inputs compare by object identity: two structurally equal tables
    built as different objects are conservatively rejected — the swap contract
    permits only ``score_policy`` to vary, so every other input must literally
    be the same object the session already owns."""
    return (
        ("model", id(provider._model)),
        ("self_cond_segment_size", int(config.self_cond_segment_size)),
        ("segment_mode", config.segment_mode),
        ("n_sae_features", int(config.n_sae_features)),
        ("num_buckets", int(config.num_buckets)),
        ("pool_mode", FeaturePredLossMode(config.pool_mode)),
        ("selection_producer", config.selection_producer),
        ("bucket_producer", config.bucket_producer),
        ("positive_count_policy", _positive_count_policy(config)),
        ("feature_mask", config.feature_mask),
        ("cluster_descriptor", _cluster_descriptor(config)),
        ("percentile_table", _percentile_table_id(config)),
        ("selection_rng", _selection_rng_id(config)),
        ("precomputed_thresholds", _precomputed_thresholds_id(config)),
        ("fp_logit_bias", _fp_logit_bias_id(config)),
    )


class ContextProvider(Protocol):
    """Cache lookup interface for real prompt/context feature segments."""

    def get(
        self,
        token_ids: Sequence[int],
    ) -> tuple[_ContextFeatureSegment, ...] | None:
        """Return feature segments for the exact token key, or None on miss."""


@dataclass(frozen=True)
class SegmentedEvalPlan:
    """Feature-slot routing for one context plus one final target segment."""

    encoder_segment_conditions: list[dict[int, int]]
    prompt_segment_to_slot: list[int]
    generation_slot: int


@dataclass(frozen=True)
class SegmentedEncoderState:
    """Precomputed packed encoder state + target routing for KV-cache generation.

    ``target_base_idx`` is the slot offset after all context segments.
    ``target_token_assignments`` is a 1-D int64 tensor covering the complete
    decoder target span. It includes forced/tool-output positions (mask=0),
    which consume KV positions even though they are omitted from scoring.
    ``assignments[t]`` is the 0-indexed target slot for absolute target-span
    offset ``t``. Decode at relative offset ``r`` uses
    ``target_base_idx + assignments[min(r, len-1)]``.
    ``prefill_end_pos`` is ``kv_cache.get_pos()`` at end of prefill, used to
    compute the relative offset during decode.

    Note: ``frozen=True`` prevents field rebinding only; tensor element
    contents remain mutable post-construction.
    """

    encoder_out: PackedEncoderOutput
    target_base_idx: int
    target_token_assignments: torch.Tensor
    prefill_end_pos: int
    target_start_pos: int | None = None


@dataclass(frozen=True)
class ContextConditioningRow:
    """Context tokens and extracted feature segments for one batch row."""

    context_tokens: list[int]
    context_segments: tuple[_ContextFeatureSegment, ...]
    context_assignments: torch.Tensor | None = None
    target_start_pos: int | None = None


@dataclass(frozen=True)
class SelfCondRuntimeSnapshot:
    state: SelfCondState
    fp_buffer: tuple[torch.Tensor, ...]
    tokens_in_seg: int


@dataclass(frozen=True)
class SelfCondPromptSeed:
    """Real prompt conditioning and S0 before the assistant marker is committed.

    This is the stable search-time boundary for fixed-surrogate evaluations.
    In turn mode, committing ``<assistant_start>`` advances the runtime to S1
    when the self-conditioning segment size is one, so a snapshot taken after
    the ordinary prefill is already too late.
    """

    state: SelfCondState
    source_logits: torch.Tensor


def seed_self_conditioning_from_prompt(
    *,
    model,
    runtime: SelfCondRuntime,
    input_ids: torch.Tensor,
    encoder_out: PackedEncoderOutput,
    seg_map: torch.Tensor,
    kv_cache,
) -> SelfCondPromptSeed:
    """Forward the real prompt and return S0 before its final marker commit."""

    source_mask = final_real_prompt_source_mask(
        seg_map,
        encoder_slot_count=encoder_out.layout.n_segments,
    )
    prefix_map = seg_map[:, :-1]
    with timed_region("selfcond/prefill/prefix_forward", sync_cuda=False):
        _prefix_logits, prefix_fp_logits = model(
            input_ids[:, :-1],
            None,
            seg_map=prefix_map,
            kv_cache=kv_cache,
            encoder_out=encoder_out,
            encoder_strict=True,
            return_fp_logits=True,
        )
    source_logits = prefix_fp_logits[:, source_mask, :].detach()
    prefill_state = SelfCondState(
        encoder_out=encoder_out.detach(),
        target_slot_idx=encoder_out.layout.n_segments - 1,
    )
    with timed_region("selfcond/prefill/provider_seed", sync_cuda=False):
        predicted_state = runtime.provider.seed_from_prompt(
            prefill_state,
            source_logits,
        )
    return SelfCondPromptSeed(state=predicted_state, source_logits=source_logits)


def _normalise_segment_conditions(
    conditions: Sequence[FeatureConditions],
) -> list[dict[int, int]]:
    """Return a copy with keys and values coerced to ints."""
    normalised: list[dict[int, int]] = []
    for idx, cond in enumerate(conditions):
        if cond is None:
            raise ValueError(
                f"Missing segment conditioning signal at index {idx}; "
                "use an explicit empty dict only when the segment has no features"
            )
        normalised.append({int(k): int(v) for k, v in cond.items()})
    return normalised


def build_multi_segment_eval_plan(
    *,
    prefix_segment_conditions: list[dict[int, int]],
    target_segment_conditions: list[dict[int, int]],
    next_segment_conditioning_offset: bool = False,
) -> SegmentedEvalPlan:
    """Build the encoder slot list + prompt routing for a multi-segment target.

    With CURRENT mode, ``len(prefix_segment_conditions) == k`` prompt segments and
    ``len(target_segment_conditions) == m`` target segments:
        slots = [seg0, ..., seg(k-1), tgt0, ..., tgt(m-1)]
        prompt-to-slot map = [0, 1, ..., k-1]
        generation_slot = k  (the FIRST target slot — the wrapper's vectorized
                              seg_map fill computes per-token slots from
                              token_assignments, not from generation_slot.)

    NEXT mode is supported only when ``len(target_segment_conditions) == 1``;
    multi-segment + NEXT is rejected at wrapper construction time.
    """
    if not target_segment_conditions:
        raise ValueError("target_segment_conditions must contain at least one segment")
    n_prompt_segments = len(prefix_segment_conditions)

    prefix = _normalise_segment_conditions(prefix_segment_conditions)
    targets = _normalise_segment_conditions(target_segment_conditions)

    if next_segment_conditioning_offset and len(targets) > 1:
        raise ValueError(
            "NEXT mode + multi-segment target should be rejected at the "
            "wrapper level; build_multi_segment_eval_plan should not see this combination"
        )

    if next_segment_conditioning_offset:
        # Single-target NEXT routing: last prompt slot is shared with the target slot.
        if n_prompt_segments == 0:
            segment_conditions = list(targets)
            prompt_to_slot: list[int] = []
            generation_slot = 0
        else:
            segment_conditions = prefix[1:] + list(targets)
            prompt_to_slot = list(range(n_prompt_segments))
            generation_slot = n_prompt_segments - 1
    else:
        segment_conditions = list(prefix) + list(targets)
        prompt_to_slot = list(range(n_prompt_segments))
        generation_slot = n_prompt_segments  # first target slot

    return SegmentedEvalPlan(
        encoder_segment_conditions=segment_conditions,
        prompt_segment_to_slot=prompt_to_slot,
        generation_slot=generation_slot,
    )


def build_segmented_feature_encoder_inputs(
    segment_conditions: list[FeatureConditions],
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded encoder inputs for multiple eval segments.

    Args:
        segment_conditions: List of per-segment ``{feature_id: bucket}`` dicts.
            Empty dict means the segment is all-masked.
        device: Target device.

    Returns:
        ``(enc_feature_ids, enc_magnitudes, enc_mask)`` with shape
        ``(1, n_seg, F_max)``. ``F_max`` is at least 1 so segments with no
        active features still produce valid tensors.
    """
    if len(segment_conditions) == 0:
        segment_conditions = [{}]

    normalised = _normalise_segment_conditions(segment_conditions)
    n_seg = len(normalised)
    F_max = max((len(cond) for cond in normalised), default=0)
    F_max = max(F_max, 1)

    enc_feature_ids = torch.zeros(1, n_seg, F_max, dtype=torch.int64, device=device)
    enc_magnitudes = torch.zeros(1, n_seg, F_max, dtype=torch.int64, device=device)
    enc_mask = torch.zeros(1, n_seg, F_max, dtype=torch.bool, device=device)

    for s_idx, cond in enumerate(normalised):
        if not cond:
            continue
        fids = list(cond.keys())
        mags = [cond[fid] for fid in fids]
        n_f = len(fids)
        enc_feature_ids[0, s_idx, :n_f] = torch.tensor(fids, dtype=torch.int64, device=device)
        enc_magnitudes[0, s_idx, :n_f] = torch.tensor(mags, dtype=torch.int64, device=device)
        enc_mask[0, s_idx, :n_f] = True

    return enc_feature_ids, enc_magnitudes, enc_mask


class TargetAwareEncoderDecoderEvalWrapper(nn.Module):
    """Encoder-decoder eval adapter with separate context and target segments.

    Observed prompt/context positions attend to real extracted context features,
    while the final scored/generated segment attends to an explicit
    operator-provided feature dictionary.

    **Lifecycle constraint:** Each prefill with a fresh KV cache
    (``kv_cache.get_pos() == 0``) overwrites ``_generation_encoder_state``.
    Subsequent decode calls reuse that prompt's state. Sequential generation
    calls may therefore reuse one wrapper, but concurrent calls may not.

    **Non-KV-cache forward paths are stateless across calls.** When the
    wrapper is invoked without a kv_cache (e.g. ``nanochat.core_eval.evaluate_core``
    calling ``forward_model`` which does not pass kv_cache), ``_generation_encoder_state``
    stays None and a single wrapper instance can be safely reused across many
    forward calls. This is the path PR-A's mid-training CORE eval uses with one
    wrapper per mode iterating over all CORE examples.

    **Thread-safety:** Not thread-safe. The wrapper holds mutable state in
    ``_generation_encoder_state``, ``_active_target_mask``, and
    ``_active_context_token_ids``. Each thread driving generation must
    build its own wrapper instance.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target: TargetConditioning,
        context_provider: ContextProvider,
        target_conditioning_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
        self_cond_config: SelfCondConfig | None = None,
        context_routing_mode: ContextRoutingMode = ContextRoutingMode.DECODER_ALIGNED,
        turn_contract: TurnContextContract | None = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if context_provider is None:
            raise TypeError(
                "TargetAwareEncoderDecoderEvalWrapper requires context_provider; "
                "pass a cache-backed provider for real prompt/context features."
            )
        self.context_provider = context_provider
        self.context_routing_mode = ContextRoutingMode(context_routing_mode)
        self.turn_contract = turn_contract
        self._prepared_turn_contexts: dict[tuple[int, ...], PreparedTurnContext] = {}
        provider_contract = getattr(context_provider, "turn_contract", None)
        if self.context_routing_mode is ContextRoutingMode.TURN_RAW_SOURCE:
            if turn_contract is None or provider_contract is None:
                raise ValueError("raw-source turn routing requires a checkpoint contract and a turn-capable provider")
            validate_turn_context_contract(provider_contract.semantic, turn_contract)
            if isinstance(target, SelfConditionedTarget) and (
                self_cond_config is None
                or self_cond_config.segment_mode is not SegmentMode.turn_delimited_fixed_tokens
                or self_cond_config.segment_mode is not SegmentMode(turn_contract.lineage.source_segment_mode)
            ):
                raise ValueError("raw-source self-conditioning requires the checkpoint turn segment mode")
            if model.dec_config.sequence_len != turn_contract.model_max_seq_len:
                raise ValueError("turn wrapper model sequence_len disagrees with checkpoint contract")
            for field in ("n_sae_features", "num_buckets"):
                if getattr(model.encoder.config, field) != getattr(turn_contract.lineage, field):
                    raise ValueError(f"turn wrapper model {field} disagrees with checkpoint contract")
            if model.feature_vocab != turn_contract.feature_vocab:
                raise ValueError("turn wrapper model feature vocabulary disagrees with checkpoint contract")
            if compute_tokenizer_checksum(tokenizer) != turn_contract.semantic["decoder_tokenizer_checksum"]:
                raise ValueError("turn wrapper decoder tokenizer disagrees with checkpoint contract")
        elif turn_contract is not None or provider_contract is not None:
            raise ValueError("turn provider/contract requires explicit raw-source turn routing")
        self.target_conditioning_mode = TargetConditioningMode.parse(target_conditioning_mode)
        self.is_target_aware_encdec = True
        self.is_self_conditioned = isinstance(target, SelfConditionedTarget)

        self.config = model.dec_config
        self.max_seq_len = model.dec_config.sequence_len
        enc_cfg = model.encoder.config

        # Self-conditioned generation is a separate code path: the wrapper
        # owns a SelfCondRuntime container with all required fields. Either
        # this is None (every other target type) or fully populated.
        self._self_cond: SelfCondRuntime | None = None
        self.self_cond_segment_mode: SegmentMode | None = None
        self._turn_specials: ChatSpecialTokens | None = None
        self._turn_structural_stop_token_ids: frozenset[int] = frozenset()

        # Normalize static targets into a list of segment dicts plus a 1-D
        # int64 token-assignment vector. SelfConditionedTarget has no static
        # target slot: prefill contains only real context segments.
        match target:
            case SelfConditionedTarget():
                if self_cond_config is None:
                    raise ValueError(
                        "SelfConditionedTarget requires a self_cond_config "
                        "argument (SelfCondConfig) describing the segment "
                        "cap, pool mode, and selection pipeline."
                    )
                if model.feature_pred_head is None:
                    raise fp_head_unavailable_error(
                        model,
                        requested="SelfConditionedTarget (self-conditioned generation)",
                        uncompacted=(
                            "SelfConditionedTarget requires a model trained with "
                            "feature_pred_config; got a model with "
                            "feature_pred_head=None."
                        ),
                    )
                provider = SelfConditionedFeatureProvider(
                    model=model,
                    cfg=self_cond_config,
                    device=next(model.parameters()).device,
                )
                self.self_cond_segment_mode = self_cond_config.segment_mode
                if self.is_turn_self_conditioned:
                    self._turn_specials = ChatSpecialTokens.from_tokenizer(tokenizer)
                    bos_id = tokenizer.get_bos_token_id()
                    if isinstance(bos_id, bool) or not isinstance(bos_id, int):
                        raise ValueError(
                            "turn-mode self-conditioned generation requires "
                            "tokenizer.get_bos_token_id() to return an int; "
                            f"got {bos_id!r}"
                        )
                    resolved_structural_ids: list[int] = [bos_id]
                    for name in _TURN_STRUCTURAL_TOKEN_NAMES:
                        token_id = tokenizer.encode_special(name)
                        if isinstance(token_id, bool) or not isinstance(token_id, int):
                            raise ValueError(
                                "turn-mode self-conditioned generation requires "
                                f"tokenizer.encode_special({name!r}) to return an int; "
                                f"got {token_id!r}"
                            )
                        resolved_structural_ids.append(token_id)
                    if len(set(resolved_structural_ids)) != len(resolved_structural_ids):
                        raise ValueError(
                            "turn-mode self-conditioned generation requires distinct "
                            "chat structural token IDs"
                        )
                    self._turn_structural_stop_token_ids = frozenset(resolved_structural_ids)
                self._self_cond = SelfCondRuntime(
                    target=target,
                    config=self_cond_config,
                    provider=provider,
                    state=None,
                    fp_buffer=(),
                    tokens_in_seg=0,
                )
                self._target_segments = []
                self._target_token_assignments = torch.empty(0, dtype=torch.long)
            case SingleTargetSegment(features=features):
                self._target_segments = [dict(features)]
                self._target_token_assignments = torch.tensor([0], dtype=torch.long)
            case MultiTargetSegments(segments=segs, token_assignments=ta):
                if self.target_conditioning_mode is TargetConditioningMode.NEXT and len(segs) > 1:
                    raise ValueError(
                        "MultiTargetSegments with len(segments) > 1 is not "
                        "supported under target_conditioning_mode=NEXT. "
                        "Lockstep + NEXT semantics are deferred. Either use "
                        "SingleTargetSegment, MultiTargetSegments with "
                        "len(segments) == 1, or set "
                        "target_conditioning_mode=CURRENT."
                    )
                self._target_segments = [dict(s) for s in segs]
                self._target_token_assignments = ta.to(dtype=torch.long)
            case _:
                raise TypeError(
                    f"target must be SingleTargetSegment, MultiTargetSegments, "
                    f"or SelfConditionedTarget; got {type(target).__name__}"
                )

        for idx, features in enumerate(self._target_segments):
            validate_feature_dict(
                features,
                enc_cfg.n_sae_features,
                enc_cfg.num_buckets,
                name=f"target segment {idx}",
                feature_vocab=None if turn_contract is None else turn_contract.feature_vocab,
            )
        self._active_target_mask = None
        self._active_context_token_ids = None
        self._generation_encoder_state = None
        # RLVR installs this only while collecting one trajectory.  Keeping the
        # seam on the already-authoritative conditioning builder avoids a
        # second prompt-boundary implementation and leaves the legacy path
        # byte-for-byte unchanged when no callback is present.
        self._self_cond_prompt_trace_callback: Callable[..., None] | None = None
        self._self_cond_boundary_trace_callback: Callable[..., None] | None = None

    def get_device(self):
        return self.model.get_device()

    @property
    def num_cache_layers(self) -> int:
        """Self-conditioning needs head history from the first prompt token."""
        return self.model.num_cache_layers if self.is_self_conditioned else self.config.n_layer

    @property
    def is_turn_self_conditioned(self) -> bool:
        """Whether this wrapper runs the turn-delimited generation contract."""

        return self.self_cond_segment_mode is SegmentMode.turn_delimited_fixed_tokens

    @property
    def generation_stop_token_ids(self) -> frozenset[int]:
        """Structural tokens that terminate ordinary turn-mode generation."""

        return self._turn_structural_stop_token_ids

    def self_cond_runtime_snapshot(self) -> SelfCondRuntimeSnapshot:
        runtime = self._self_cond
        if runtime is None:
            raise RuntimeError(
                "self_cond_runtime_snapshot is available only in self-conditioned mode"
            )
        state = runtime.state
        if state is None:
            raise RuntimeError("self_cond_runtime_snapshot cannot run before prefill")
        return SelfCondRuntimeSnapshot(
            state=state,
            fp_buffer=runtime.fp_buffer,
            tokens_in_seg=runtime.tokens_in_seg,
        )

    def self_cond_segment_size(self) -> int:
        """Return the immutable decode-window size for self-conditioned mode."""
        runtime = self._self_cond
        if runtime is None:
            raise RuntimeError("self_cond_segment_size is available only in self-conditioned mode")
        return int(runtime.config.self_cond_segment_size)

    def restore_self_cond_runtime(self, snapshot: SelfCondRuntimeSnapshot) -> None:
        runtime = self._self_cond
        if runtime is None:
            raise RuntimeError(
                "restore_self_cond_runtime is available only in self-conditioned mode"
            )
        if runtime.state is None:
            raise RuntimeError("restore_self_cond_runtime cannot run before prefill")

        _require_nonempty_self_cond_state(
            snapshot.state,
            owner="restore_self_cond_runtime",
        )
        tokens_in_seg = snapshot.tokens_in_seg
        segment_size = int(runtime.config.self_cond_segment_size)
        if type(tokens_in_seg) is not int or not (0 <= tokens_in_seg < segment_size):
            raise ValueError(
                "restore_self_cond_runtime tokens_in_seg must be an int excluding bool "
                f"in [0, {segment_size}), got {tokens_in_seg!r}"
            )
        if len(snapshot.fp_buffer) != tokens_in_seg:
            raise ValueError(
                "restore_self_cond_runtime fp_buffer cardinality must equal "
                f"tokens_in_seg; got {len(snapshot.fp_buffer)} and {tokens_in_seg}"
            )

        runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = (
            snapshot.state,
            tuple(snapshot.fp_buffer),
            tokens_in_seg,
        )

    def swap_self_cond_provider(
        self,
        provider: SelfConditionedFeatureProvider,
        config: SelfCondConfig,
    ) -> None:
        runtime = self._self_cond
        if runtime is None:
            raise RuntimeError("swap_self_cond_provider is available only in self-conditioned mode")
        if provider.cfg is not config:
            raise ValueError(
                "swap_self_cond_provider requires provider.cfg to be the supplied "
                "config object; the compatibility key and the runtime must "
                "describe the same configuration"
            )

        old_key = _self_cond_compat_key(runtime.provider, runtime.config)
        new_key = _self_cond_compat_key(provider, config)
        for (old_name, old_value), (new_name, new_value) in zip(
            old_key,
            new_key,
            strict=True,
        ):
            assert old_name == new_name
            if old_value != new_value:
                raise ValueError(
                    "swap_self_cond_provider incompatible component "
                    f"{old_name}: old={old_value!r}, new={new_value!r}"
                )

        runtime.provider, runtime.config = provider, config

    @contextmanager
    def conditioning_context(self, target_mask=None, context_token_ids=None):
        """Temporarily set target positions and context token sequences."""

        prev_target_mask = self._active_target_mask
        prev_context_token_ids = self._active_context_token_ids
        self._active_target_mask = target_mask
        self._active_context_token_ids = context_token_ids
        try:
            yield self
        finally:
            self._active_target_mask = prev_target_mask
            self._active_context_token_ids = prev_context_token_ids

    def forward(self, input_ids, targets=None, kv_cache=None, loss_reduction="mean"):
        with timed_region("encdec_eval_wrapper_forward"):
            if self._self_cond is not None:
                return self._forward_self_cond(
                    input_ids,
                    targets=targets,
                    kv_cache=kv_cache,
                    loss_reduction=loss_reduction,
                )

            encoder_out, seg_map = self._build_conditioning(
                input_ids,
                kv_cache=kv_cache,
            )

            return self.model(
                input_ids,
                targets,
                seg_map=seg_map,
                loss_reduction=loss_reduction,
                kv_cache=kv_cache,
                encoder_out=encoder_out,
                encoder_strict=True,
            )

    def _forward_self_cond(
        self,
        input_ids,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
    ):
        """Self-conditioned forward path.

        Both modes forward the real prompt prefix and pool exactly the FP rows
        routed to its final real source slot to construct S0.  Fixed mode keeps
        the historical two-phase boundary replay: the assistant-boundary token
        is outside its first generated window, and each Kth generated input is
        committed under S1 before its next-token distribution is returned.
        Turn mode instead makes ``<assistant_start>`` row one of the first
        assistant window.  Every input is committed once under its outgoing
        state; the Kth input's outgoing logits are retained while S1 is
        published for the following input.
        """
        if targets is not None:
            raise ValueError(
                "Self-conditioned forward does not support targets "
                "(loss-only scoring); it is purely a generation-time path."
            )
        if kv_cache is None:
            raise RuntimeError(
                "Self-conditioned forward requires a kv_cache (it is "
                "purely a generation-time path). Loss-only forwards must "
                "use a non-self-cond target."
            )
        runtime = self._self_cond
        assert runtime is not None  # narrowed by caller branch

        B, T = input_ids.size()
        if B != 1:
            raise ValueError(f"Self-conditioned generation supports B=1 only; got B={B}")
        if kv_cache.get_pos() == 0:
            return self._self_cond_prefill(input_ids, runtime, kv_cache)
        return self._self_cond_decode_step(input_ids, runtime, kv_cache, input_ids.device)

    def _self_cond_prompt_seed(
        self,
        input_ids,
        runtime: SelfCondRuntime,
        kv_cache,
    ) -> SelfCondPromptSeed:
        """Build real prompt conditioning and seed S0 before committing its marker."""

        with timed_region("selfcond/prefill/build_conditioning", sync_cuda=False):
            encoder_out, seg_map = self._build_conditioning(
                input_ids,
                kv_cache=kv_cache,
            )
        B, _T = input_ids.shape
        if B != 1:
            raise ValueError("self-conditioned prefill requires B=1")
        if self.is_turn_self_conditioned:
            assert self._turn_specials is not None  # set in __init__ for turn mode
            marker_id = int(input_ids[0, -1])
            if marker_id != self._turn_specials.assistant_start:
                raise ValueError(
                    "turn-mode self-conditioned prefill requires input_ids to end in "
                    "the same <assistant_start> marker validated on the context "
                    f"tokens ({self._turn_specials.assistant_start}); got {marker_id}"
                )
        return seed_self_conditioning_from_prompt(
            model=self.model,
            runtime=runtime,
            input_ids=input_ids,
            encoder_out=encoder_out,
            seg_map=seg_map,
            kv_cache=kv_cache,
        )

    def _self_cond_prefill(
        self,
        input_ids,
        runtime: SelfCondRuntime,
        kv_cache,
    ):
        """Seed S0 from the final real prompt slot, then commit the last token.

        Both modes pool exactly the FP rows routed to the final real prompt
        slot; they differ only in what the terminal prompt token is.  Under
        ``fixed_tokens`` it is the assistant boundary, committed under the
        predicted state and outside its first generated window.  Under
        ``turn_delimited_fixed_tokens`` it is ``<assistant_start>``, a real
        first-assistant input whose FP row is window row one, so the shared
        decode step commits it and publishes exactly the atomic runtime tuple
        decode would — which is why K=1 and K>1 behave identically here.
        """

        prompt_seed = self._self_cond_prompt_seed(input_ids, runtime, kv_cache)
        predicted_state = prompt_seed.state
        if self.is_turn_self_conditioned:
            if int(runtime.config.self_cond_segment_size) == 1:
                boundary_trace_callback = self._self_cond_boundary_trace_callback
                if boundary_trace_callback is not None:
                    boundary_trace_callback(
                        committed_prefix=tuple(int(token) for token in input_ids[0].tolist()),
                        pool_positions=(int(input_ids.shape[1]) - 1,),
                        first_committed_position=int(input_ids.shape[1]),
                        source="assistant",
                    )
            marker_result = self_conditioned_decode_step(
                model=self.model,
                provider=runtime.provider,
                state=predicted_state,
                fp_buffer=(),
                tokens_in_seg=0,
                emitted_token=input_ids[:, -1:],
                kv_cache=kv_cache,
                device=input_ids.device,
            )
            runtime.state = marker_result.state
            runtime.fp_buffer = marker_result.fp_buffer
            runtime.tokens_in_seg = marker_result.tokens_in_seg
            return marker_result.logits

        final_seg_map = torch.full(
            (1, 1),
            predicted_state.target_slot_idx,
            dtype=torch.long,
            device=input_ids.device,
        )
        with timed_region("selfcond/prefill/boundary_forward", sync_cuda=False):
            logits = self.model(
                input_ids[:, -1:],
                None,
                seg_map=final_seg_map,
                kv_cache=kv_cache,
                encoder_out=predicted_state.encoder_out.to(input_ids.device),
                encoder_strict=True,
            )
        runtime.state = predicted_state
        runtime.fp_buffer = ()
        runtime.tokens_in_seg = 0
        return logits

    def _self_cond_decode_step(
        self,
        input_ids,
        runtime: SelfCondRuntime,
        kv_cache,
        device,
    ):
        """Delegate one emitted token to the shared causal transition."""
        B, T = input_ids.size()
        if (B, T) != (1, 1):
            raise ValueError(
                "self-conditioned decode requires one emitted token with shape "
                f"(1, 1), got {(B, T)}"
            )
        state = runtime.state
        assert state is not None  # prefill must have run before decode
        if self.is_turn_self_conditioned:
            token_id = int(input_ids.item())
            if token_id in self._turn_structural_stop_token_ids:
                raise ValueError(
                    "turn-mode self-conditioned generation received an unexpected "
                    f"structural token id {token_id}; ordinary assistant generation "
                    "must stop before forwarding chat/tool structure"
                )
        result = self_conditioned_decode_step(
            model=self.model,
            provider=runtime.provider,
            state=state,
            fp_buffer=runtime.fp_buffer,
            tokens_in_seg=runtime.tokens_in_seg,
            emitted_token=input_ids,
            kv_cache=kv_cache,
            device=device,
        )
        runtime.state = result.state
        runtime.fp_buffer = result.fp_buffer
        runtime.tokens_in_seg = result.tokens_in_seg
        return result.logits

    def _build_conditioning(self, input_ids, kv_cache=None):
        with timed_region("encdec_build_conditioning"):
            return self._build_conditioning_impl(input_ids, kv_cache=kv_cache)

    def _build_conditioning_impl(self, input_ids, kv_cache=None):
        B, T = input_ids.size()
        device = input_ids.device
        self._validate_active_context(B, T)
        if self.context_routing_mode is ContextRoutingMode.TURN_RAW_SOURCE and B != 1:
            raise ValueError("turn target-real wrapper requires batch size 1")

        if kv_cache is not None and kv_cache.get_pos() > 0:
            if self._generation_encoder_state is None:
                raise RuntimeError(
                    "Decode called before a target-aware encoder prefill state was built"
                )
            state = self._generation_encoder_state
            if state.target_start_pos is not None:
                positions = torch.arange(kv_cache.get_pos(), kv_cache.get_pos() + T, device=device)
                assignments = state.target_token_assignments.to(device=device)
                relative = (positions - state.target_start_pos).clamp(0, assignments.numel() - 1)
                seg_map = (state.target_base_idx + assignments[relative]).unsqueeze(0)
                return repeat_packed_encoder_output(state.encoder_out, B), seg_map
            rel = kv_cache.get_pos() - state.prefill_end_pos
            rel_clamped = min(
                max(0, int(rel)),
                state.target_token_assignments.numel() - 1,
            )
            target_idx = state.target_base_idx + int(
                state.target_token_assignments[rel_clamped].item()
            )
            seg_map = torch.full(
                (B, T),
                target_idx,
                dtype=torch.long,
                device=device,
            )
            return (
                repeat_packed_encoder_output(state.encoder_out, B),
                seg_map,
            )

        row_payloads: list[ContextConditioningRow] = []
        for row in range(B):
            lookup_tokens = self._context_tokens_for_row(
                input_ids[row],
                row=row,
                kv_cache=kv_cache,
            )
            if self.context_routing_mode is ContextRoutingMode.TURN_RAW_SOURCE:
                key = tuple(lookup_tokens)
                if key not in self._prepared_turn_contexts:
                    if self.turn_contract is None:
                        raise ValueError("turn context routing requires a checkpoint-derived turn contract")
                    self._prepared_turn_contexts[key] = prepare_turn_context(
                        lookup_tokens, self._lookup_context_segments(lookup_tokens), self.tokenizer,
                        source_bos_char_len=self.turn_contract.semantic["source_bos_char_len"],
                    )
                prepared = self._prepared_turn_contexts[key]
                if T < len(key) or input_ids[row, :len(key)].tolist() != list(key):
                    raise ValueError("turn wrapper input must begin with the exact completion-prefix key")
                row_payloads.append(ContextConditioningRow(
                    context_tokens=lookup_tokens[:-1], context_segments=prepared.segments,
                    context_assignments=prepared.token_assignments, target_start_pos=prepared.target_start_pos,
                ))
                continue
            context_segments = self._lookup_context_segments(lookup_tokens)
            # Align BEFORE the encoder pack so encoder slots map 1:1 to
            # post-alignment context segments. See _align_row_context_segments
            # for rationale (dead-encoder-slot fix).
            context_segments = self._align_row_context_segments(
                lookup_tokens,
                context_segments,
            )
            context_tokens = lookup_tokens
            if self.is_turn_self_conditioned:
                context_tokens, context_segments = self._prepare_turn_self_cond_prompt_context(
                    lookup_tokens,
                    context_segments,
                )
            row_payloads.append(
                ContextConditioningRow(
                    context_tokens=context_tokens,
                    context_segments=context_segments,
                )
            )

        enc_feature_ids, enc_magnitudes, enc_mask = self._pack_encoder_inputs(
            row_payloads,
            device=device,
        )

        # The model owns mixed precision explicitly: CUDA embeddings and
        # activations use COMPUTE_DTYPE while Linear casts its master weights
        # to the incoming activation dtype.  An autocast context here is both
        # redundant and unsafe: newer PyTorch releases may promote the final
        # encoder RMSNorm to fp32, leaving cross-attention with bf16 queries
        # and fp32 keys/values.
        with torch.no_grad():
            encoder_out = self.model.encoder(
                enc_feature_ids,
                enc_magnitudes,
                enc_mask,
            )

        seg_map = torch.zeros(B, T, dtype=torch.long, device=device)
        target_base_indices: list[int] = []
        for row, payload in enumerate(row_payloads):
            plan = self._plan_for_context(payload.context_segments)
            ctx_count = len(plan.encoder_segment_conditions) - len(self._target_segments)
            target_base_idx = ctx_count
            target_base_indices.append(target_base_idx)

            if payload.context_assignments is not None:
                seg_map[row, :len(payload.context_tokens)] = payload.context_assignments.to(device=device)
            else:
                self._fill_context_assignments(
                    seg_map[row],
                    payload.context_tokens,
                    payload.context_segments,
                )

            final_start = (payload.target_start_pos if payload.target_start_pos is not None
                           else self._final_segment_start(row=row, T=T, kv_cache=kv_cache))
            if not self.is_self_conditioned and final_start is not None:
                target_span_len = T - final_start
                rel = torch.arange(target_span_len, device=device)
                rel_clamped = rel.clamp_max(self._target_token_assignments.numel() - 1)
                seg_map[row, final_start:] = (
                    target_base_idx + self._target_token_assignments.to(device=device)[rel_clamped]
                )

        prompt_trace_callback = self._self_cond_prompt_trace_callback
        if prompt_trace_callback is not None:
            if B != 1 or kv_cache is None or kv_cache.get_pos() != 0:
                raise RuntimeError("RLVR prompt tracing requires one fresh batch-one prefill")
            prompt_trace_callback(
                input_ids=input_ids,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
                encoder_slot_count=encoder_out.layout.n_segments,
            )

        if not self.is_self_conditioned and kv_cache is not None and kv_cache.get_pos() == 0:
            if B != 1:
                raise ValueError("Target-aware generation prefill expects batch size 1")
            self._generation_encoder_state = SegmentedEncoderState(
                encoder_out=encoder_out.detach(),
                target_base_idx=int(target_base_indices[0]),
                target_token_assignments=self._target_token_assignments.detach().clone(),
                prefill_end_pos=int(kv_cache.get_pos() + T),
                target_start_pos=row_payloads[0].target_start_pos,
            )

        return encoder_out, seg_map

    def _plan_for_context(
        self,
        context_segments: tuple[_ContextFeatureSegment, ...],
    ) -> SegmentedEvalPlan:
        prefix_conditions = [
            self._context_segment_features(segment) for segment in context_segments
        ]
        if self.is_self_conditioned:
            for segment_index, features in enumerate(prefix_conditions):
                if not features:
                    raise EmptySelfConditioningSegmentError(
                        "self-conditioned prefill requires every real context "
                        f"segment to contain at least one feature; segment "
                        f"{segment_index} is empty"
                    )
            return SegmentedEvalPlan(
                encoder_segment_conditions=prefix_conditions,
                prompt_segment_to_slot=list(range(len(prefix_conditions))),
                generation_slot=len(prefix_conditions) - 1,
            )
        return build_multi_segment_eval_plan(
            prefix_segment_conditions=prefix_conditions,
            target_segment_conditions=self._target_segments,
            next_segment_conditioning_offset=(
                self.target_conditioning_mode is TargetConditioningMode.NEXT
            ),
        )

    def _validate_active_context(self, batch_size: int, seq_len: int) -> None:
        if self._active_target_mask is not None:
            target_shape = tuple(self._active_target_mask.shape)
            if target_shape != (batch_size, seq_len):
                raise ValueError(
                    "target_mask must have shape matching input_ids; "
                    f"got {target_shape}, expected {(batch_size, seq_len)}"
                )
        if (
            self._active_context_token_ids is not None
            and len(self._active_context_token_ids) != batch_size
        ):
            raise ValueError(
                "context_token_ids must contain one token sequence per batch row; "
                f"got {len(self._active_context_token_ids)}, expected {batch_size}"
            )
        if self._active_context_token_ids is not None:
            for row, context_tokens in enumerate(self._active_context_token_ids):
                context_length = len(context_tokens)
                if context_length < 1:
                    raise ValueError(
                        "each context_token_ids row must contain at least one "
                        f"token; row {row} has {context_length}"
                    )

    def _context_tokens_for_row(self, row_input_ids, *, row: int, kv_cache=None):
        if self._active_context_token_ids is not None:
            return [int(tok) for tok in self._active_context_token_ids[row]]

        tokens = [int(tok) for tok in row_input_ids.tolist()]
        if kv_cache is not None and kv_cache.get_pos() == 0:
            return tokens

        target_mask = self._target_mask_for_row(row)
        if target_mask is not None and target_mask.any():
            first_target_pos = int(target_mask.nonzero(as_tuple=True)[0][0].item())
            return tokens[: first_target_pos + 1]

        return tokens

    def _target_mask_for_row(self, row: int):
        if self._active_target_mask is None:
            return None
        target_mask = self._active_target_mask
        if target_mask.device != self.get_device():
            target_mask = target_mask.to(self.get_device())
        return target_mask[row].bool()

    def _prepare_turn_self_cond_prompt_context(
        self,
        context_tokens: list[int],
        context_segments: tuple[_ContextFeatureSegment, ...],
    ) -> tuple[list[int], tuple[_ContextFeatureSegment, ...]]:
        """Remove exactly one terminal marker-only real feature segment.

        The full rendered prompt remains the context-cache lookup key.  Once its
        extracted turn geometry is validated, the terminal assistant marker's
        real SAE slot is discarded and the marker token is withheld from real
        routing.  Turn prefill later forwards that marker under the predicted
        first-assistant state and retains its FP row.
        """

        specials = self._turn_specials
        if specials is None:
            raise RuntimeError("turn-mode self-conditioned prompt specials are unavailable")
        spans = legacy_turn_spans(context_tokens, specials)
        # ``legacy_turn_spans`` always closes its final span at the last token,
        # and only ``<assistant_start>`` opens an ASSISTANT span, so the span's
        # start position plus the terminal token id decide marker-only-ness.
        # Interior turn starts are NOT required to be window starts: a Gemma
        # token can straddle an unmarked boundary (tool output into assistant
        # text), and training routes the straddled first token with the
        # preceding window; the builder reproduces that ownership exactly.
        if (
            spans[-1].start != len(context_tokens) - 2
            or context_tokens[-1] != specials.assistant_start
        ):
            raise ValueError(
                "turn-mode self-conditioned prefill requires an exact rendered "
                "conversation ending in one marker-only <assistant_start> turn"
            )

        offsets = token_to_char_offsets(self.tokenizer, context_tokens)
        marker_start = offsets[-2]
        marker_end = offsets[-1]
        marker_segment = context_segments[-1]
        if (
            int(marker_segment.char_start) != marker_start
            or int(marker_segment.char_end) != marker_end
        ):
            raise ValueError(
                "turn-mode self-conditioned context cache must end in the exact "
                "marker-only <assistant_start> feature segment: expected character "
                f"range [{marker_start}, {marker_end}), got "
                f"[{marker_segment.char_start}, {marker_segment.char_end})"
            )

        real_segments = context_segments[:-1]
        reach = int(real_segments[-1].char_end) if real_segments else None
        if reach != marker_start:
            raise ValueError(
                "turn-mode self-conditioned real prompt features must end exactly "
                f"where <assistant_start> begins ({marker_start}); got {reach}"
            )
        if getattr(self.context_provider, "segment_mode", None) is not self.self_cond_segment_mode:
            raise ValueError("turn-mode prefill requires context produced in the same segment mode")
        assignments = getattr(self.context_provider, "token_assignments", None)
        if assignments is None or len(assignments) != len(context_tokens):
            raise ValueError(
                "turn-mode prefill requires persisted token ownership for the exact prompt"
            )
        if any(index < 0 or index >= len(context_segments) for index in assignments):
            raise ValueError("turn-mode prompt token ownership contains an invalid segment index")
        if assignments[-1] != len(context_segments) - 1 or any(
            index == len(context_segments) - 1 for index in assignments[:-1]
        ):
            raise ValueError(
                "turn-mode terminal feature segment must own only the assistant marker"
            )

        return context_tokens[:-1], real_segments

    def _final_segment_start(self, *, row: int, T: int, kv_cache=None) -> int | None:
        if kv_cache is not None and kv_cache.get_pos() == 0:
            return max(T - 1, 0)

        # Target-aware callers already supply the exact prompt/context prefix,
        # including the decoder position that predicts the first target token.
        # Keep that routing boundary on the host instead of rediscovering it
        # with ``target_mask.any()/nonzero().item()`` on every CUDA batch row.
        if self._active_context_token_ids is not None:
            return len(self._active_context_token_ids[row]) - 1

        target_mask = self._target_mask_for_row(row)
        if target_mask is not None and target_mask.any():
            return int(target_mask.nonzero(as_tuple=True)[0][0].item())

        return max(T - 1, 0) if T > 0 else None

    def _lookup_context_segments(
        self,
        context_tokens: list[int],
    ) -> tuple[_ContextFeatureSegment, ...]:
        try:
            segments = self.context_provider.get(context_tokens)
        except Exception as e:
            raise RuntimeError(
                f"context_provider.get() failed for "
                f"{token_context_key(context_tokens)} "
                f"(context token count={len(context_tokens)}). "
                f"This usually indicates cache corruption or I/O error. "
                f"Original error: {type(e).__name__}: {e}\n\n"
                f"{context_cache_generation_instructions()}"
            ) from e
        if segments is None:
            raise KeyError(
                "No context features found for "
                f"{token_context_key(context_tokens)} "
                f"(context token count={len(context_tokens)}). "
                "The manifest must contain one row for the exact nanochat "
                "token sequence used by this eval prompt/context.\n\n"
                f"{context_cache_generation_instructions()}"
            )
        if len(segments) == 0:
            raise KeyError(
                "Context cache returned no segments for "
                f"{token_context_key(context_tokens)}. Rebuild the cache with "
                "full prompt/context feature extraction.\n\n"
                f"{context_cache_generation_instructions()}"
            )
        return segments

    def _pack_encoder_inputs(self, row_payloads, *, device):
        n_seg_max = max(
            self._num_encoder_segments_for_context(payload.context_segments)
            for payload in row_payloads
        )
        f_max = max(
            [
                max((len(d) for d in self._target_segments), default=0),
                1,
                *[
                    len(segment.feature_ids)
                    for payload in row_payloads
                    for segment in payload.context_segments
                ],
            ]
        )

        B = len(row_payloads)
        # Context features arrive as CPU tuples/dicts. Assemble the complete
        # padded batch on CPU, then transfer each tensor once; writing every
        # segment directly into CUDA storage turns thousands of tiny source
        # tensors and slice assignments into synchronous host/device copies.
        enc_feature_ids = torch.zeros(B, n_seg_max, f_max, dtype=torch.long)
        enc_magnitudes = torch.zeros(B, n_seg_max, f_max, dtype=torch.long)
        enc_mask = torch.zeros(B, n_seg_max, f_max, dtype=torch.bool)

        enc_cfg = self.model.encoder.config
        feature_vocab = None if self.turn_contract is None else self.turn_contract.feature_vocab
        for row, payload in enumerate(row_payloads):
            for seg_idx, features in enumerate(
                self._encoder_feature_slots(payload.context_segments)
            ):
                validate_feature_dict(
                    features,
                    enc_cfg.n_sae_features,
                    enc_cfg.num_buckets,
                    name=f"encoder segment {seg_idx}",
                    feature_vocab=feature_vocab,
                )
                self._write_feature_slot(
                    enc_feature_ids,
                    enc_magnitudes,
                    enc_mask,
                    row,
                    seg_idx,
                    features,
                )

        return (
            enc_feature_ids.to(device=device, non_blocking=True),
            enc_magnitudes.to(device=device, non_blocking=True),
            enc_mask.to(device=device, non_blocking=True),
        )

    def _num_encoder_segments_for_context(
        self,
        segments: tuple[_ContextFeatureSegment, ...],
    ) -> int:
        return len(self._plan_for_context(segments).encoder_segment_conditions)

    def _encoder_feature_slots(
        self,
        segments: tuple[_ContextFeatureSegment, ...],
    ) -> list[dict[int, int]]:
        return self._plan_for_context(segments).encoder_segment_conditions

    @staticmethod
    def _context_segment_features(segment: _ContextFeatureSegment) -> dict[int, int]:
        if len(segment.feature_ids) != len(segment.buckets):
            raise ValueError(
                "context segment: feature_ids and buckets must have the same "
                f"length, got {len(segment.feature_ids)} and {len(segment.buckets)}"
            )
        if len(set(segment.feature_ids)) != len(segment.feature_ids):
            raise ValueError(
                "context segment: duplicate feature ids are not valid in eval context caches"
            )
        return {
            int(fid): int(bucket)
            for fid, bucket in zip(segment.feature_ids, segment.buckets, strict=True)
        }

    @staticmethod
    def _write_feature_slot(
        enc_feature_ids,
        enc_magnitudes,
        enc_mask,
        row: int,
        seg_idx: int,
        features: dict[int, int],
    ) -> None:
        if not features:
            return
        fids = list(features.keys())
        buckets = [features[fid] for fid in fids]
        n_features = len(fids)
        enc_feature_ids[row, seg_idx, :n_features] = torch.tensor(
            fids,
            dtype=torch.long,
            device=enc_feature_ids.device,
        )
        enc_magnitudes[row, seg_idx, :n_features] = torch.tensor(
            buckets,
            dtype=torch.long,
            device=enc_magnitudes.device,
        )
        enc_mask[row, seg_idx, :n_features] = True

    def _align_row_context_segments(
        self,
        context_tokens: list[int],
        context_segments: tuple[_ContextFeatureSegment, ...],
    ) -> tuple[_ContextFeatureSegment, ...]:
        """Align ``context_segments`` to the consumer tokenizer's decoded
        prefix length BEFORE the encoder is packed.

        The lifted alignment fixes the dead-encoder-slot pattern: with
        alignment INSIDE ``_fill_context_assignments`` (the pre-lift
        position), ``_pack_encoder_inputs`` packed unaligned segments into
        N encoder slots; if alignment filtered tail segments to M < N, the
        encoder's [M, N) slots were populated but never referenced by any
        seg_map value. The lifted alignment shrinks the slot count to
        match.

        Small deltas (<= ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS``) are
        absorbed; larger deltas raise ``ValueError`` and the row is not
        packed.

        Args:
            context_tokens: nanochat-tokenized prefix tokens.
            context_segments: unaligned ``ContextFeatureSegment`` tuple
                from the context provider (or a dummy single-segment).

        Returns:
            Aligned ``ContextFeatureSegment`` tuple. When no alignment
            change is needed, returns ``context_segments`` unchanged
            (object-identical).
        """
        if not context_tokens or not context_segments:
            return context_segments
        return align_segments_to_decoded_prefix(
            tokenizer=self.tokenizer,
            prefix_tokens=context_tokens,
            segments=context_segments,
            counter_name=_CONTEXT_ALIGNMENT_COUNTER,
        )

    def _fill_context_assignments(
        self,
        row_seg_map,
        context_tokens: list[int],
        context_segments: tuple[_ContextFeatureSegment, ...],
    ) -> None:
        """Fill ``row_seg_map[:n_prefix]`` with cummax-monotone segment
        assignments derived from per-token / per-segment char overlap.

        Turn mode consumes the exact ownership persisted by extraction, after
        the prefill boundary validated its producing mode and terminal marker.

        **Pre-condition:** ``context_segments`` MUST already be aligned to
        ``context_tokens``'s decoded length. Alignment is performed
        upstream by ``_align_row_context_segments`` inside
        ``_build_conditioning_impl`` so the encoder slot count matches.
        Callers that bypass ``_build_conditioning_impl`` (e.g. unit tests
        that bind this method directly to a stub) must align upstream.
        """
        if not context_tokens:
            return

        persisted = (
            getattr(self.context_provider, "token_assignments", None)
            if self.is_turn_self_conditioned
            else None
        )
        if self.is_turn_self_conditioned and persisted is None:
            raise ValueError("turn-mode context requires persisted token ownership")
        assignments = context_token_assignments(
            segment_mode=self.self_cond_segment_mode or SegmentMode.fixed_tokens,
            tokenizer=self.tokenizer,
            context_tokens=context_tokens,
            context_segments=context_segments,
            token_assignments=None if persisted is None else persisted[: len(context_tokens)],
        )
        row_seg_map[: len(context_tokens)] = assignments.to(row_seg_map.device)
