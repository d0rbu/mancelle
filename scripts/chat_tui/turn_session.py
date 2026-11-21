"""Transactional conversation epochs over the shared one-shot turn runtime.

Only the worker-owned candidate is mutable during generation. The committed
driver, its KV allocation and its RNGs stay alive until a successful swap.
Streaming events are previews, not publication of a half-built epoch.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch as t

from nanochat.sampling import sample_next_token
from scripts.chat_tui.session import (
    BoundarySnapshot,
    ChatSession,
    DiscardPreview,
    GenerationEvent,
    GenerationEventKind,
    LedgerSegment,
    SegmentKind,
    SegmentProvenance,
)
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondState,
)
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.target_conditioning import SelfConditionedTarget, SingleEntryContextProvider
from steering.segmentation import SegmentMode
from steering.sft.turn_spans import ChatSpecialTokens, TurnRole, legacy_turn_spans

EventSink = Callable[[GenerationEvent], None]
CancelCheck = Callable[[], bool]


class StaleTurnOperation(RuntimeError):
    """An operation no longer owns the conversation it started against."""


class _CancelledTurn(RuntimeError):
    """Cooperative cancellation discards the candidate, including RNG draws."""


@dataclass(frozen=True)
class _AppendedSlot:
    slot: int
    features: dict[int, int]
    overridden: bool


class _LedgerProvider(SelfConditionedFeatureProvider):
    """Observe appends without changing the core selection or decode algorithm.

    Prefill can append both S0 and S1 when K=1. ``last_appended_segment`` alone
    cannot describe that ledger, so retain both append receipts until consumed.
    """

    def __init__(
        self, model, cfg, device, *, prefill_overrides: Mapping[int, _ExplicitSegment]
    ) -> None:
        super().__init__(model, cfg, device)
        self.receipts: list[_AppendedSlot] = []
        self._prefill_overrides = dict(prefill_overrides)
        self._append_ordinal = 0

    def _append_from_fp_logits(
        self, state: SelfCondState, fp_logits_tail: t.Tensor, *, owner: str, max_time: int | None
    ) -> SelfCondState:
        overridden = self._queued_explicit is not None
        explicit = self._prefill_overrides.pop(self._append_ordinal, None)
        if explicit is not None:
            self.queue_explicit_segment(explicit.features)
            overridden = explicit.overridden
        result = super()._append_from_fp_logits(
            state, fp_logits_tail, owner=owner, max_time=max_time
        )
        assert self.last_appended_segment is not None
        self.receipts.append(
            _AppendedSlot(result.target_slot_idx, dict(self.last_appended_segment), overridden)
        )
        self._append_ordinal += 1
        return result


@dataclass(frozen=True)
class _ExplicitSegment:
    """A prefill window forwarded with fixed features instead of a prediction."""

    features: dict[int, int]
    overridden: bool  # ledger provenance: an operator edit, or a pinned committed window


_NO_OVERRIDES: Mapping[int, _ExplicitSegment] = MappingProxyType({})


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[int, ...]:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("turn Chat requires a complete user message before the assistant marker")
    # render_for_completion has a historical 2048-token truncation default.
    # Use the same renderer with its explicit unbounded limit, then append the
    # single marker, exactly as that helper does after rendering the messages.
    tokens, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=2**31)
    return (*tokens, int(tokenizer.encode_special("<assistant_start>")))


class _TurnDriver(ChatSession):
    """Private, single-owner epoch; fixed-mode mutation entrypoints are unused."""

    def __init__(self, *, epoch_id: int, revision: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epoch_id = epoch_id
        self.revision = revision
        self.messages: list[dict[str, Any]] = []
        self.prompt_messages: list[dict[str, Any]] = []
        self.prefill_rng: tuple[t.Tensor, t.Tensor] | None = None
        self.prepared_context: PreparedPromptContext | None = None

    def _build_self_cond_config(self, policy) -> SelfCondConfig:
        return replace(
            super()._build_self_cond_config(policy),
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )

    def _validate_snapshot_ordinal(self, ordinal: int) -> None:
        if type(ordinal) is not int:
            raise ValueError("turn snapshot ordinal must be an integer excluding bool")
        super()._validate_snapshot_ordinal(ordinal)

    def fork(self, *, epoch_id: int, revision: int, copy_runtime: bool) -> _TurnDriver:
        candidate = _TurnDriver(
            setup=self._setup,
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            config=self._config,
            extraction=self._extraction,
            epoch_id=epoch_id,
            revision=revision,
        )
        candidate._active_steering = self._active_steering
        candidate._score_policy = self._score_policy
        candidate._attempt = self._attempt
        candidate._token_rng.set_state(self._token_rng.get_state())
        candidate._selection_rng.set_state(self._selection_rng.get_state())
        candidate._conversation.restore_tokens(self._conversation.tokens)
        candidate.messages = copy.deepcopy(self.messages)
        candidate.prompt_messages = copy.deepcopy(self.prompt_messages)
        candidate.prefill_rng = self.prefill_rng
        candidate.prepared_context = self.prepared_context
        candidate._prefill_len = self._prefill_len
        candidate._aligned_segments = self._aligned_segments
        if copy_runtime:
            if self._wrapper is None or self._kv_cache is None:
                raise RuntimeError("cannot fork an epoch before prefill")
            config = candidate._build_self_cond_config(candidate._score_policy)
            # A wrapper has one mutable runtime; parameters and immutable packed
            # encoder snapshots are shared, never KV buffers or RNG generators.
            candidate._wrapper = copy.copy(self._wrapper)
            assert self._wrapper._self_cond is not None
            candidate._wrapper._self_cond = replace(
                self._wrapper._self_cond,
                config=config,
                provider=_LedgerProvider(
                    self._model, config, self._device, prefill_overrides=_NO_OVERRIDES
                ),
            )
            candidate._kv_cache = candidate._make_kv_cache()
            candidate._kv_cache.prefill(self._kv_cache)
            candidate._snapshots = list(self._snapshots)
            candidate._ledger = [
                replace(entry, epoch_id=epoch_id, revision=revision) for entry in self._ledger
            ]
        return candidate

    def _record_snapshot(self, ordinal: int, logits: t.Tensor) -> None:
        assert self._wrapper is not None and self._kv_cache is not None
        runtime = self._wrapper.self_cond_runtime_snapshot()
        position = int(self._kv_cache.get_pos())
        if position != len(self._conversation.tokens):
            raise RuntimeError("turn snapshot KV position differs from committed input rows")
        # S0 is a prefill restart anchor. With K=1 its completed marker has
        # already published S1; callers restart prefill instead of restoring
        # ordinal 0 as though the marker were outside the generated window.
        self._snapshots.append(
            BoundarySnapshot(
                selfcond_ordinal=ordinal,
                kv_pos=position,
                runtime=runtime,
                next_logits=logits.detach().to(device="cpu").clone(),
                token_rng_state=self._token_rng.get_state().clone(),
                selection_rng_state=self._selection_rng.get_state().clone(),
                token_count=position,
                ledger_len=len(self._ledger),
            )
        )

    def _append_receipt(self, receipt: _AppendedSlot, ordinal: int, start: int) -> None:
        stamp = self._steering_stamp()
        self._ledger.append(
            LedgerSegment(
                kind=SegmentKind.SELFCOND,
                encoder_slot_idx=receipt.slot,
                selfcond_ordinal=ordinal,
                token_start=start,
                token_end=start,
                features=dict(receipt.features),
                provenance=(
                    SegmentProvenance.OVERRIDDEN
                    if receipt.overridden
                    else SegmentProvenance.STEERED
                    if stamp is not None
                    else SegmentProvenance.PREDICTED
                ),
                steering=stamp,
                color_idx=self._next_color(),
                epoch_id=self.epoch_id,
                revision=self.revision,
            )
        )

    def prefill(
        self,
        messages,
        on_event: EventSink,
        *,
        overrides: Mapping[int, _ExplicitSegment] = _NO_OVERRIDES,
        prepared: PreparedPromptContext | None = None,
    ) -> t.Tensor:
        prompt = _render_prompt(self._tokenizer, messages)
        required = len(prompt) + self._config.max_new_tokens_per_turn + 1
        limit = int(self._model.dec_config.sequence_len)
        if required > limit:
            raise ValueError(
                f"turn Chat requires {required} RustBPE tokens "
                f"({len(prompt)} prompt + {self._config.max_new_tokens_per_turn} assistant "
                f"+ 1 closing marker), exceeding target sequence_len={limit}; no truncation"
            )
        if prepared is None:
            prepared = self._extraction.prepare(prompt)
        if prepared.token_ids != prompt or prepared.assistant_end_id != self._assistant_end_id:
            raise ValueError("turn extraction does not match the exact rendered prompt")
        if prepared.segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
            raise ValueError("turn extraction must carry turn-delimited segment mode")
        if prepared.token_assignments is None:
            raise ValueError("turn extraction must carry persisted token ownership")
        aligned = prepared.segments
        config = self._build_self_cond_config(self._score_policy)
        provider = _LedgerProvider(self._model, config, self._device, prefill_overrides=overrides)
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=self._model,
            tokenizer=self._tokenizer,
            target=SelfConditionedTarget(),
            context_provider=SingleEntryContextProvider(
                prompt,
                aligned,
                segment_mode=prepared.segment_mode,
                token_assignments=prepared.token_assignments,
            ),
            self_cond_config=config,
        )
        wrapper.swap_self_cond_provider(provider, config)
        cache = self._make_kv_cache()
        self.prefill_rng = (
            self._token_rng.get_state().clone(),
            self._selection_rng.get_state().clone(),
        )
        with wrapper.conditioning_context(context_token_ids=[list(prompt)]):
            logits = wrapper(t.tensor([prompt], device=self._device), kv_cache=cache)[:, -1, :]
        self._conversation.restore_tokens(prompt)
        self.prompt_messages = copy.deepcopy(messages)
        self.prepared_context = prepared
        self._wrapper, self._kv_cache = wrapper, cache
        self._prefill_len, self._aligned_segments = len(prompt), aligned
        self._ledger, self._snapshots = [], []
        # The authoritative converter includes a marker-only slot; the wrapper
        # has validated and removed it from encoder packing. The fixed ledger
        # helper can validate that same empty terminal display span, then drop it.
        self._append_prefix_ledger(
            prompt, aligned, assignments=t.tensor(prepared.token_assignments, dtype=t.long)
        )
        marker = self._ledger.pop()
        if marker.token_start != marker.token_end or marker.token_start != len(prompt) - 1:
            raise RuntimeError("turn prefix ledger terminal slot is not marker-only")
        self._ledger = [
            replace(entry, epoch_id=self.epoch_id, revision=self.revision) for entry in self._ledger
        ]
        receipts, provider.receipts = provider.receipts, []
        expected = 2 if self._config.self_cond_segment_size == 1 else 1
        if len(receipts) != expected:
            raise RuntimeError("turn prefill append count disagrees with checkpoint cadence")
        self._append_receipt(receipts[0], 0, len(prompt) - 1)
        self._record_snapshot(0, logits)
        if len(receipts) == 2:
            self._ledger[-1] = replace(self._ledger[-1], token_end=len(prompt))
            self._append_receipt(receipts[1], 1, len(prompt))
            self._record_snapshot(1, logits)
        self._emit_transcript(on_event)
        return logits

    def _handle_boundary(self, logits, on_event, provenance_override=None) -> None:
        del provenance_override
        provider = self._runtime_provider()
        receipts, provider.receipts = provider.receipts, []
        if not receipts:
            return
        if len(receipts) != 1:
            raise RuntimeError("one turn decode input appended more than one slot")
        previous = self._ledger[-1]
        assert previous.selfcond_ordinal is not None
        boundary = len(self._conversation.tokens)
        self._ledger[-1] = replace(previous, token_end=boundary)
        ordinal = previous.selfcond_ordinal + 1
        self._append_receipt(receipts[0], ordinal, boundary)
        self._record_snapshot(ordinal, logits)
        on_event(GenerationEvent(GenerationEventKind.SEGMENT_APPENDED, None, str(ordinal)))

    def generate(self, logits: t.Tensor, on_event: EventSink, cancelled: CancelCheck) -> None:
        assert self._wrapper is not None and self._prefill_len is not None
        stops = self._wrapper.generation_stop_token_ids
        already_generated = len(self._conversation.tokens) - self._prefill_len
        remaining = self._config.max_new_tokens_per_turn - already_generated
        for index in range(remaining):
            if cancelled():
                raise _CancelledTurn
            token = int(
                sample_next_token(
                    logits, self._token_rng, self._config.temperature, self._config.top_k
                ).item()
            )
            if token in stops:
                break
            # Match InferenceEngine exactly: a budget-ending ordinary token is
            # displayed but never forwarded, since it predicts no later sample.
            if index + 1 == remaining:
                delta = self._conversation.append_tokens([token])
            else:
                logits, delta = self._commit_token(token, on_event)
            on_event(GenerationEvent(GenerationEventKind.TOKEN, delta, str(token)))
        if cancelled():
            raise _CancelledTurn
        response = self._conversation.tokens[self._prefill_len :]
        self.messages = [
            *copy.deepcopy(self.prompt_messages),
            {"role": "assistant", "content": self._tokenizer.decode(list(response))},
        ]
        # Logical closing is not another model input. All sampled structural
        # tokens, including tool markers, were stopped before display/forward.
        delta = self._conversation.append_tokens([self._assistant_end_id])
        on_event(GenerationEvent(GenerationEventKind.TOKEN, delta, str(self._assistant_end_id)))
        on_event(GenerationEvent(GenerationEventKind.TURN_CLOSED, None, ""))
        self.transcript_spans()  # prove complete ledger tiling before publication


class TurnChatSession:
    """Atomic facade used by the TUI; each successful operation swaps one owner."""

    def __init__(self, *, setup, config, **kwargs) -> None:
        cadence = setup.turn_self_cond_segment_size_rustbpe
        if setup.segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
            raise ValueError("TurnChatSession requires a turn-delimited checkpoint")
        if type(cadence) is not int or cadence <= 0:
            raise ValueError("turn Chat requires positive integer checkpoint RustBPE cadence")
        self._driver = _TurnDriver(
            setup=setup,
            config=replace(config, self_cond_segment_size=cadence),
            epoch_id=0,
            revision=0,
            **kwargs,
        )
        self._start_messages: list[dict[str, Any]] = []
        self._revision = 0
        self._operation = 0
        self._active_identity: tuple[int, int] | None = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> tuple[int, int]:
        return self._driver.epoch_id, self._revision

    def assert_current_entry(self, entry: LedgerSegment) -> None:
        if (entry.epoch_id, entry.revision) != self.identity:
            raise StaleTurnOperation(
                "selected ledger segment belongs to an earlier Chat epoch/revision"
            )
        if not any(
            (row.kind, row.encoder_slot_idx, row.selfcond_ordinal)
            == (entry.kind, entry.encoder_slot_idx, entry.selfcond_ordinal)
            for row in self._driver.ledger()
        ):
            raise StaleTurnOperation("selected ledger segment is no longer live")

    def accepts_event(self, event: GenerationEvent) -> bool:
        return (event.epoch_id, event.revision) in (self.identity, self._active_identity)

    @t.inference_mode()
    def _run(self, build, on_event: EventSink, cancel_check, *, new_epoch: bool) -> None:
        with self._lock:
            self._operation += 1
            operation, revision = self._operation, self._revision
            old = self._driver
            # Attempt IDs are never reused, even when a transaction fails.
            # Otherwise a late preview from a failed attempt could match retry.
            event_identity = old.epoch_id + int(new_epoch), operation
            self._active_identity = event_identity

        def cancelled() -> bool:
            return (
                operation != self._operation
                or revision != self._revision
                or self._active_identity != event_identity
                or (cancel_check is not None and cancel_check())
            )

        def emit(event: GenerationEvent) -> None:
            if cancelled():
                raise _CancelledTurn
            on_event(replace(event, epoch_id=event_identity[0], revision=event_identity[1]))

        try:
            if cancelled():
                raise _CancelledTurn
            candidate, logits = build(old, event_identity[0], operation, emit)
            candidate.generate(logits, emit, cancelled)
            with self._lock:
                if cancelled():
                    raise _CancelledTurn
                self._driver = candidate
                self._revision = operation
                self._active_identity = None
        except _CancelledTurn:
            if operation == self._operation and revision == self._revision:
                on_event(
                    GenerationEvent(
                        GenerationEventKind.TRANSCRIPT, None, old._conversation.text, *self.identity
                    )
                )
            raise _CancelledTurn("turn operation cancelled; committed epoch unchanged") from None
        except Exception as exc:
            if operation == self._operation and revision == self._revision:
                on_event(
                    GenerationEvent(
                        GenerationEventKind.TRANSCRIPT, None, old._conversation.text, *self.identity
                    )
                )
                on_event(
                    GenerationEvent(
                        GenerationEventKind.ERROR_ROLLBACK,
                        None,
                        f"turn operation discarded; committed epoch unchanged: {exc}",
                        *self.identity,
                    )
                )
            raise
        finally:
            if self._active_identity == event_identity:
                self._active_identity = None

    def cancel(self) -> None:
        with self._lock:
            # Keep the attempt ID so its owner can restore the visible committed
            # transcript. A newer attempt increments it and suppresses that old
            # rollback, even if the newer attempt itself failed to commit.
            self._active_identity = None

    def load_prefix(self, messages: list) -> None:
        owned = copy.deepcopy(messages)
        if owned and owned[-1].get("role") != "assistant":
            raise ValueError("turn Chat conversation prefix must end in a completed assistant turn")
        with self._lock:
            old = self._driver
            candidate = old.fork(
                epoch_id=old.epoch_id + 1, revision=self._operation + 1, copy_runtime=False
            )
            tokens = (
                old._tokenizer.render_conversation({"messages": owned}, max_tokens=2**31)[0]
                if owned
                else [old._tokenizer.get_bos_token_id()]
            )
            if len(tokens) >= int(old._model.dec_config.sequence_len):
                raise ValueError("turn Chat prefix exceeds the target RustBPE context window")
            candidate._conversation.restore_tokens(tuple(tokens))
            candidate._discard_runtime()
            candidate.messages = owned
            candidate.prompt_messages = []
            candidate.prefill_rng = None
            candidate.prepared_context = None
            candidate._attempt = 0
            candidate._reseed_generators()
            self._start_messages = copy.deepcopy(owned)
            self._driver = candidate
            self._operation += 1
            self._revision = self._operation
            self._active_identity = None

    def clear(self) -> None:
        self.load_prefix(self._start_messages)

    def set_seed(self, seed: int) -> None:
        with self._lock:
            self._driver.set_seed(seed)
            self._invalidate_controls()

    def _invalidate_controls(self) -> None:
        self._operation += 1
        self._revision = self._operation
        self._driver.revision = self._revision
        self._driver._ledger = [
            replace(row, revision=self._revision) for row in self._driver._ledger
        ]
        self._active_identity = None

    def set_steering(self, selection) -> None:
        with self._lock:
            # Provider replacement belongs to the next candidate, not the
            # committed runtime. Validate the policy on a runtime-free fork.
            old = self._driver
            candidate = old.fork(epoch_id=old.epoch_id, revision=self._revision, copy_runtime=False)
            candidate.set_steering(selection)
            old._active_steering = candidate._active_steering
            old._score_policy = candidate._score_policy
            self._invalidate_controls()

    def apply_steering(self, selection, on_event, *, cancel_check=None) -> None:
        """Validate, regenerate and publish steering as one turn transaction."""
        if self._driver._wrapper is None:
            self.set_steering(selection)
            return

        def build(old, epoch, revision, emit):
            candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=False)
            candidate.set_steering(selection)
            return self._restart(candidate, epoch, revision, emit)

        self._run(build, on_event, cancel_check, new_epoch=False)

    def send_user_turn(self, text, on_event, *, cancel_check=None) -> None:
        def build(old, epoch, revision, emit):
            candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=False)
            messages = [*copy.deepcopy(old.messages), {"role": "user", "content": text}]
            return candidate, candidate.prefill(messages, emit)

        self._run(build, on_event, cancel_check, new_epoch=True)

    def _restart(self, old, epoch, revision, emit, *, overrides=_NO_OVERRIDES):
        if old.prepared_context is None or old.prefill_rng is None:
            raise RuntimeError("cannot regenerate before a turn has been generated")
        candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=False)
        if candidate._config.deterministic_rewind:
            candidate._token_rng.set_state(old.prefill_rng[0])
            candidate._selection_rng.set_state(old.prefill_rng[1])
        else:
            candidate._restore_rng(None, fresh_take=True)
        return candidate, candidate.prefill(
            old.prompt_messages, emit, overrides=overrides, prepared=old.prepared_context
        )

    def rewind(self, ordinal, on_event, *, cancel_check=None) -> None:
        def build(old, epoch, revision, emit):
            old._validate_snapshot_ordinal(ordinal)
            if ordinal == 0:
                return self._restart(old, epoch, revision, emit)
            candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=True)
            logits = candidate._restore_to_snapshot(ordinal, fresh_take=True)
            candidate._emit_transcript(emit)
            return candidate, logits

        self._run(build, on_event, cancel_check, new_epoch=False)

    def modify(self, ordinal, features, on_event, *, cancel_check=None) -> None:
        def build(old, epoch, revision, emit):
            old._validate_snapshot_ordinal(ordinal)
            validated = old._runtime_provider().validate_explicit_segment(features)
            if ordinal == 0 or (ordinal == 1 and old._config.self_cond_segment_size == 1):
                overrides = {ordinal: _ExplicitSegment(dict(validated), True)}
                if ordinal == 1:
                    # Prefill appends S0 and S1 together at K=1. Forward the
                    # committed S0 as-is; re-predicting it would silently
                    # change the window this edit did not target.
                    s0 = next(row for row in old._ledger if row.selfcond_ordinal == 0)
                    overrides[0] = _ExplicitSegment(
                        dict(s0.features), s0.provenance is SegmentProvenance.OVERRIDDEN
                    )
                candidate, logits = self._restart(old, epoch, revision, emit, overrides=overrides)
            else:
                previous = old._snapshots[ordinal - 1]
                target = old._snapshots[ordinal]
                replay = old._conversation.tokens[previous.token_count : target.token_count]
                if not replay:
                    raise RuntimeError("turn edit has no outgoing input rows to replay")
                candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=True)
                logits = candidate._restore_to_snapshot(ordinal - 1, fresh_take=True)
                for index, token in enumerate(replay):
                    if index + 1 == len(replay):
                        candidate._runtime_provider().queue_explicit_segment(validated)
                    logits, _ = candidate._commit_token(token, lambda _event: None)
                candidate._emit_transcript(emit)
            edited = next(
                (row for row in candidate._ledger if row.selfcond_ordinal == ordinal), None
            )
            if (
                edited is None
                or edited.features != dict(validated)
                or edited.provenance is not SegmentProvenance.OVERRIDDEN
            ):
                landed = (
                    "absent" if edited is None else f"{edited.provenance.value} {edited.features!r}"
                )
                raise RuntimeError(f"turn edit did not land on segment {ordinal}: {landed}")
            return candidate, logits

        self._run(build, on_event, cancel_check, new_epoch=False)

    def resteer(self, on_event, *, cancel_check=None) -> None:
        self._run(self._restart, on_event, cancel_check, new_epoch=False)

    def _messages_for_prefix(self, ledger_index: int) -> list[dict[str, Any]]:
        old = self._driver
        entries = old.ledger()
        if (
            type(ledger_index) is not int
            or not 0 <= ledger_index < len(entries)
            or entries[ledger_index].kind is not SegmentKind.PREFIX_REAL
        ):
            raise ValueError("turn rewind requires a real prefix ledger entry")
        position = entries[ledger_index].token_start
        spans = legacy_turn_spans(
            old._conversation.tokens, ChatSpecialTokens.from_tokenizer(old._tokenizer)
        )
        starts = [span.start + 1 for span in spans if span.role == TurnRole.USER]
        users = [index for index, message in enumerate(old.messages) if message["role"] == "user"]
        if len(starts) != len(users):
            raise ValueError("logical transcript and rendered user-turn boundaries disagree")
        chosen = max((index for index, start in enumerate(starts) if start <= position), default=0)
        return copy.deepcopy(old.messages[: users[chosen] + 1])

    def rewind_to_prefix(self, ledger_index, on_event, *, cancel_check=None) -> None:
        messages = self._messages_for_prefix(ledger_index)

        def build(old, epoch, revision, emit):
            candidate = old.fork(epoch_id=epoch, revision=revision, copy_runtime=False)
            candidate._restore_rng(None, fresh_take=True)
            return candidate, candidate.prefill(messages, emit)

        self._run(build, on_event, cancel_check, new_epoch=True)

    def modify_prefix(self, ledger_index, features, on_event, *, cancel_check=None) -> None:
        raise ValueError(
            "turn Chat real history is read-only; edit a current-assistant predicted segment"
        )

    def discard_preview(self, entry, *, is_modify=False) -> DiscardPreview:
        self.assert_current_entry(entry)
        if entry.kind is SegmentKind.PREFIX_REAL:
            if is_modify:
                raise ValueError("turn Chat real history is read-only")
            index = next(
                i
                for i, row in enumerate(self.ledger())
                if row.encoder_slot_idx == entry.encoder_slot_idx
            )
            retained = len(
                _render_prompt(self._driver._tokenizer, self._messages_for_prefix(index))
            )
        else:
            assert entry.selfcond_ordinal is not None
            retained = self._driver._snapshots[entry.selfcond_ordinal].token_count
        return DiscardPreview(
            retained,
            max(0, len(self.conversation_tokens()) - retained),
            sum(row.token_start >= retained for row in self.ledger()),
            "modify" if is_modify else "rewind",
        )

    def ledger(self):
        return self._driver.ledger()

    def token_budget(self):
        return self._driver.token_budget()

    def conversation_tokens(self):
        return self._driver.conversation_tokens()

    def transcript_spans(self):
        return self._driver.transcript_spans()

    def open_user_text(self):
        return None

    def turn_state(self):
        return self._driver.turn_state()
