"""Scalar ChatCORE self-conditioning with the nanochat calculator transcript.

Tool source windows are planned against the current causal prefix plus the
known forced block. Only that block's ownership is consumed; committed KV is
never rerouted when the source tokenizer later sees resumed assistant text.
Assistant cadence remains the checkpoint's measured RustBPE approximation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch as t

from nanochat.engine import KVCache, RowState, use_calculator
from nanochat.sampling import sample_next_token
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.conditional.encoder_decoder.full_sequence_routing import (
    EMPTY_FEATURES,
    build_full_sequence_segment_routing,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    self_conditioned_decode_step,
)
from steering.eval.inference_engine import InferenceEngine
from steering.eval.turn_context_contract import TurnContextContract
from steering.refined_types import PositiveInt
from steering.segmentation import (
    turn_segments_for_document,
)
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    TurnRole,
    rendered_text_and_turn_char_starts,
)

__all__ = ["ChatSelfConditionedGeneration", "generate_self_conditioned_chat"]


@dataclass(frozen=True)
class ChatSelfConditionedGeneration:
    """Full transcript and sampled/forced mask, for the scalar calculator protocol."""

    token_ids: tuple[int, ...]
    mask: tuple[int, ...]
    stop_reason: str = "budget"


def plan_tool_windows(
    prefix: list[int],
    forced_tokens: tuple[int, ...],
    *,
    tokenizer: Any,
    extraction_tokenizer: Any,
    contract: TurnContextContract,
) -> tuple[tuple[int, ...], ...]:
    """Map known tool inputs through the checkpoint's source segmentation.

    Empty feature arrays below request geometry only from the shared routing
    helper. They never enter a cache, encoder, or conditioning admission path;
    every live feature slot is selected from actual model FP rows.
    """
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    if len(forced_tokens) < 2 or forced_tokens[0] != specials.output_start or forced_tokens[-1] != specials.output_end:
        raise ValueError("tool plan must contain a complete output_start/output_end block")
    ids = [*prefix, *forced_tokens]
    text, starts = rendered_text_and_turn_char_starts(ids, tokenizer)
    source_ids = list(extraction_tokenizer.encode(text))
    _segments, char_starts, char_ends, _drift = turn_segments_for_document(
        source_ids, extraction_tokenizer, text, starts,
        PositiveInt.parse(contract.lineage.source_segment_size),
    )
    routing = build_full_sequence_segment_routing(
        ids=ids, tokenizer=tokenizer,
        seg_features=([EMPTY_FEATURES] * len(char_starts), [EMPTY_FEATURES] * len(char_starts),
                      char_starts, char_ends),
        source_bos_char_len=contract.semantic["source_bos_char_len"],
    )
    # Discard prefix assignments even if future source tokenization changed
    # them. A straddle may give the first tool input the previous source ordinal,
    # but the role boundary still closes the assistant FP buffer first.
    assignments = (routing.seg_map[len(prefix):] - 1).tolist()
    if any(slot < 0 for slot in assignments):
        raise ValueError("tool plan contains unrouted decoder inputs")
    plans: list[tuple[int, ...]] = []
    start = 0
    for end in range(1, len(forced_tokens) + 1):
        if end == len(forced_tokens) or assignments[end] != assignments[start]:
            plans.append(forced_tokens[start:end])
            start = end
    return tuple(plans)


class TurnSelfConditionedTransition:
    """Consume assistant K windows and source-owned calculator windows."""

    def __init__(
        self,
        wrapper: TargetAwareEncoderDecoderEvalWrapper,
        extraction_tokenizer: Any,
        trace_callback: Any = None,
    ):
        if wrapper.turn_contract is None or wrapper._self_cond is None or extraction_tokenizer is None:
            raise ValueError("chat turn transitions require a runtime and source tokenizer contract")
        self.wrapper = wrapper
        self.runtime = wrapper._self_cond
        self.contract = wrapper.turn_contract
        self.extraction_tokenizer = extraction_tokenizer
        self.specials = ChatSpecialTokens.from_tokenizer(wrapper.tokenizer)
        self.python_ids = frozenset(wrapper.tokenizer.encode_special(name) for name in ("<python_start>", "<python_end>"))
        self.role = TurnRole.ASSISTANT
        self.plans: deque[tuple[int, ...]] = deque()
        self.tool_rows: list[t.Tensor] = []
        self.trace_callback = trace_callback

    def plan_output(self, prefix: list[int], forced_tokens: tuple[int, ...]) -> None:
        self.plans.extend(plan_tool_windows(
            prefix, forced_tokens, tokenizer=self.wrapper.tokenizer,
            extraction_tokenizer=self.extraction_tokenizer, contract=self.contract,
        ))

    def commit(
        self,
        token: int,
        kv_cache: KVCache,
        *,
        committed_prefix: tuple[int, ...] | None = None,
    ) -> t.Tensor:
        runtime, device = self.runtime, self.wrapper.get_device()
        state = runtime.state
        if state is None:
            raise RuntimeError("chat turn transitions require a completed prefill")
        ids = t.tensor([[token]], dtype=t.long, device=device)
        entering_tool = token == self.specials.output_start and self.role is TurnRole.ASSISTANT
        if entering_tool and not self.plans:
            raise ValueError("tool input requires a known calculator output plan")
        if entering_tool or (self.role is TurnRole.TOOL and self.plans):
            plan = self.plans[0]
            if token != plan[len(self.tool_rows)]:
                raise ValueError("tool input disagrees with its calculator output")
            if entering_tool and runtime.fp_buffer:
                if self.trace_callback is not None:
                    if committed_prefix is None:
                        raise RuntimeError("traced tool transition requires its committed prefix")
                    prefix_length = len(committed_prefix)
                    self.trace_callback.prepare_boundary(
                        committed_prefix=committed_prefix[:-1],
                        pool_positions=tuple(range(
                            prefix_length - 1 - runtime.tokens_in_seg,
                            prefix_length - 1,
                        )),
                        first_committed_position=prefix_length - 1,
                        source="assistant",
                    )
                state = runtime.provider.step(state, t.cat(runtime.fp_buffer, dim=1))
            logits, fp = self.wrapper.model(
                ids, None, seg_map=t.full((1, 1), state.target_slot_idx, device=device, dtype=t.long),
                kv_cache=kv_cache, encoder_out=state.encoder_out.to(device),
                encoder_strict=True, return_fp_logits=True,
            )
            self.tool_rows.append(fp.detach())
            if len(self.tool_rows) == len(plan):
                if self.trace_callback is not None:
                    if committed_prefix is None:
                        raise RuntimeError("traced tool window requires its committed prefix")
                    prefix_length = len(committed_prefix)
                    self.trace_callback.prepare_boundary(
                        committed_prefix=committed_prefix,
                        pool_positions=tuple(range(
                            prefix_length - len(self.tool_rows),
                            prefix_length,
                        )),
                        first_committed_position=prefix_length,
                        source="tool",
                    )
                state = runtime.provider.append_tool_window(state, t.cat(self.tool_rows, dim=1))
                self.tool_rows.clear()
                self.plans.popleft()
            runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = state, (), 0
            self.role = TurnRole.TOOL
            return logits
        if token in self.wrapper.generation_stop_token_ids and token not in self.python_ids:
            raise ValueError("unexpected structural input in the assistant turn")
        completes_window = runtime.tokens_in_seg + 1 == int(runtime.config.self_cond_segment_size)
        if completes_window and self.trace_callback is not None:
            if committed_prefix is None:
                raise RuntimeError("traced assistant boundary requires its committed prefix")
            prefix_length = len(committed_prefix)
            self.trace_callback.prepare_boundary(
                committed_prefix=committed_prefix,
                pool_positions=tuple(range(
                    prefix_length - runtime.tokens_in_seg - 1,
                    prefix_length,
                )),
                first_committed_position=prefix_length,
                source="assistant",
            )
        result = self_conditioned_decode_step(
            model=self.wrapper.model, provider=runtime.provider, state=state,
            fp_buffer=runtime.fp_buffer, tokens_in_seg=runtime.tokens_in_seg,
            emitted_token=ids, kv_cache=kv_cache, device=device,
        )
        runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = result.state, result.fp_buffer, result.tokens_in_seg
        self.role = TurnRole.ASSISTANT
        return result.logits


def generate_self_conditioned_chat(
    engine: InferenceEngine,
    token_ids: list[int],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 1234,
    extraction_tokenizer: Any = None,
    trace_callback: Any = None,
) -> ChatSelfConditionedGeneration:
    """Generate a scalar full transcript, charging sampled and forced outputs.

    With N>=1 budgeted nonterminal outputs, appends including S0 are
    ``1 + N//K`` (turn) and ``1 + (N-1)//K`` (fixed). A terminal sampled
    after N outputs needs one more input evaluation: ``1 + (N+1)//K`` and
    ``1 + N//K``. Terminal tokens and the final budgeted input never commit.
    """
    # A caller may itself be in inference mode. RLVR's saved prompt tensors
    # must remain ordinary tensors so differentiable replay can consume them.
    with t.inference_mode(trace_callback is None), t.no_grad():
        wrapper = engine._model
        if not isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper) or not wrapper.is_self_conditioned:
            raise ValueError("scalar chat requires a self-conditioned wrapper")
        if wrapper._self_cond is None or wrapper._self_cond.state is not None:
            raise ValueError("scalar chat requires a fresh self-conditioned runtime per example")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        if len(token_ids) < 2:
            raise ValueError("self-conditioned chat requires at least two prompt tokens")
        engine._validate_prompt_length(len(token_ids))
        tokenizer = engine._tokenizer
        if token_ids[-1] != tokenizer.encode_special("<assistant_start>"):
            raise ValueError("self-conditioned chat prompt must end in assistant_start")
        budget = min(max_tokens, wrapper.config.sequence_len - len(token_ids))
        if budget == 0:
            return ChatSelfConditionedGeneration(
                tuple(token_ids),
                (0,) * len(token_ids),
                "budget",
            )
        transition = (
            TurnSelfConditionedTransition(wrapper, extraction_tokenizer, trace_callback)
            if wrapper.is_turn_self_conditioned
            else None
        )
        prefill = engine.prepare_self_conditioned_prefill(token_ids, max_new_tokens=budget)
        logits, kv_cache = prefill.logits, prefill.kv_cache
        rng = t.Generator(device=engine._device).manual_seed(seed)
        row = RowState(token_ids.copy())
        masks = [0] * len(token_ids)
        python_start, python_end, output_start, output_end, assistant_end = (
            tokenizer.encode_special(name) for name in
            ("<python_start>", "<python_end>", "<output_start>", "<output_end>", "<assistant_end>")
        )
        terminal_ids = {assistant_end, tokenizer.get_bos_token_id()}
        stop_reason = "budget"
        for index in range(budget):
            # Match the scalar calculator's sampling stream, even for forced rows.
            sampled = int(sample_next_token(logits, rng, temperature, top_k).item())
            forced = bool(row.forced_tokens)
            if not forced and trace_callback is not None:
                trace_callback.record_nonforced_logits_prefix_length(len(row.current_tokens))
            token = row.forced_tokens.popleft() if forced else sampled
            if token in terminal_ids:
                stop_reason = "eos"
                break
            row.current_tokens.append(token)
            masks.append(0 if forced else 1)
            if (
                trace_callback is not None
                and not forced
                and transition is not None
                and token in wrapper.generation_stop_token_ids
                and token not in terminal_ids
                and (
                    token not in transition.python_ids
                    or (token == python_start and row.in_python_block)
                    or (token == python_end and not row.in_python_block)
                )
            ):
                trace_callback.record_invalid_structural_token(token)
                stop_reason = "invalid_structure"
                break
            if token == python_start:
                row.in_python_block, row.python_expr_tokens = True, []
            elif token == python_end and row.in_python_block:
                row.in_python_block = False
                if row.python_expr_tokens:
                    result = use_calculator(tokenizer.decode(row.python_expr_tokens))
                    if result is not None:
                        output = (output_start, *tokenizer.encode(str(result)), output_end)
                        row.forced_tokens.extend(output)
                        if transition is not None and index + 1 < budget:
                            transition.plan_output(row.current_tokens, output)
                row.python_expr_tokens = []
            elif row.in_python_block:
                row.python_expr_tokens.append(token)
            if index + 1 == budget:
                break
            if transition is None:
                logits = wrapper(t.tensor([[token]], dtype=t.long, device=engine._device), kv_cache=kv_cache)
            else:
                logits = transition.commit(
                    token,
                    kv_cache,
                    committed_prefix=tuple(row.current_tokens),
                )
            logits = logits[:, -1, :]
        if trace_callback is not None and transition is not None and stop_reason != "invalid_structure":
            # Keep native budget/EOS accounting. A delimiter sentinel records why
            # the retained transcript is nevertheless invalid for RLVR reward.
            if row.in_python_block:
                trace_callback.record_invalid_structural_token(python_start)
            elif row.forced_tokens:
                trace_callback.record_invalid_structural_token(output_start)
        return ChatSelfConditionedGeneration(
            tuple(row.current_tokens),
            tuple(masks),
            stop_reason,
        )
