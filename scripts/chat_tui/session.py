"""Conversation state and continuous-stream driver for the chat TUI."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import Any

import torch as t
from phantom.sized import NonEmptyStr

from nanochat.engine import KVCache
from nanochat.sampling import sample_next_token
from scripts.chat_tui.convo import assert_renderable
from scripts.chat_tui.extraction import ExtractionService
from scripts.chat_tui.steering import (
    SteeringMethod,
    SteeringSelection,
    build_score_policy,
)
from scripts.completion_tui.color_manager import next_color
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    SelfCondRuntimeSnapshot,
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    selection_without_delta,
)
from steering.conditional.feature_pipeline import active_bucket_count
from steering.conditional.sampling import SamplingBundle
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SelfConditionedTarget,
    SingleEntryContextProvider,
)
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    align_segments_to_decoded_prefix,
    compute_context_token_segment_assignments,
    token_to_char_offsets,
)

_CHAT_TUI_ALIGNMENT_COUNTER = NonEmptyStr.parse("segmentation.char_end_extended.chat_tui")


class TurnState(StrEnum):
    BETWEEN = "between"
    IN_USER = "in_user"
    IN_ASSISTANT = "in_assistant"


@dataclass(frozen=True)
class ChatSessionConfig:
    self_cond_segment_size: int
    temperature: float
    top_k: int | None
    seed: int
    max_new_tokens_per_turn: int
    deterministic_rewind: bool

    def __post_init__(self) -> None:
        positive_ints = {
            "self_cond_segment_size": self.self_cond_segment_size,
            "seed": self.seed,
            "max_new_tokens_per_turn": self.max_new_tokens_per_turn,
        }
        if self.top_k is not None:
            positive_ints["top_k"] = self.top_k
        for field, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive int, got {value!r}")

        finite_floats = {"temperature": self.temperature}
        for field, value in finite_floats.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field} must be finite, got {value!r}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature!r}")


def build_session_self_cond_config(
    setup: Any,
    config: ChatSessionConfig,
    selection_rng: t.Generator,
    policy: Any,
    *,
    segment_mode: SegmentMode,
) -> SelfCondConfig:
    """Build the driver-owned self-conditioning config for a chat session."""

    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(int(config.self_cond_segment_size)),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            # Pretrain-shaped setups carry delta_weights on their output
            # selection (the leaf path pre-strips it); live self-conditioning
            # cannot replay delta weighting, so strip it here exactly like
            # axbench's build_self_conditioning_config does.
            selection=selection_without_delta(setup.output_selection),
            percentile_table=setup.percentile_table,
            rng=selection_rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        # Chat setup and target-model loading both pin the encoder to two
        # buckets, one of which is reserved for inactive features.
        num_buckets=PositiveInt.parse(active_bucket_count(2, reserves_negative_bucket=True)),
        n_sae_features=PositiveInt.parse(int(setup.n_sae_features)),
        feature_mask=setup.feature_mask,
        cluster_consolidation=setup.consolidation,
        score_policy=policy,
        segment_mode=segment_mode,
    )


@dataclass(frozen=True)
class TextDelta:
    appended: str
    rewritten_from: int | None


class SegmentKind(StrEnum):
    PREFIX_REAL = "prefix_real"
    SELFCOND = "selfcond"


class SegmentProvenance(StrEnum):
    REAL = "real"
    PREDICTED = "predicted"
    STEERED = "steered"
    OVERRIDDEN = "overridden"


@dataclass(frozen=True)
class SteeringStamp:
    method: SteeringMethod
    signal_names: tuple[str, ...]
    total_strength: float


@dataclass(frozen=True)
class LedgerSegment:
    kind: SegmentKind
    encoder_slot_idx: int
    selfcond_ordinal: int | None
    token_start: int
    token_end: int
    features: dict[int, int]
    provenance: SegmentProvenance
    steering: SteeringStamp | None
    color_idx: int
    # Fixed streams retain their historical identity. Turn Chat binds every
    # row to the committed conversation epoch/revision that created it.
    epoch_id: int = 0
    revision: int = 0


@dataclass(frozen=True)
class DiscardPreview:
    retained_tokens: int
    discarded_tokens: int
    discarded_segments: int
    operation: str


@dataclass(frozen=True)
class TranscriptSpan:
    char_start: int
    char_end: int
    color_idx: int
    kind: SegmentKind
    is_live: bool


@dataclass(frozen=True)
class BoundarySnapshot:
    selfcond_ordinal: int
    kv_pos: int
    runtime: SelfCondRuntimeSnapshot
    next_logits: t.Tensor
    token_rng_state: t.Tensor
    selection_rng_state: t.Tensor
    token_count: int
    ledger_len: int


@dataclass(frozen=True)
class _PrefillWorld:
    """A fully-forwarded wrapper/KV pair not yet installed into the session."""

    wrapper: TargetAwareEncoderDecoderEvalWrapper
    kv_cache: Any
    logits: t.Tensor
    s0_overridden: bool


@dataclass(frozen=True)
class _RngEntryState:
    """Both generator states plus the fresh-take counter at operation entry."""

    token_state: t.Tensor
    selection_state: t.Tensor
    attempt: int


class GenerationEventKind(StrEnum):
    TOKEN = "token"
    SEGMENT_APPENDED = "segment_appended"
    TURN_CLOSED = "turn_closed"
    ERROR_ROLLBACK = "error_rollback"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True)
class GenerationEvent:
    kind: GenerationEventKind
    text_delta: TextDelta | None
    detail: str
    epoch_id: int | None = None
    revision: int | None = None


class ConversationText:
    """Conversation tokens and their progressively decoded transcript."""

    def __init__(self, tokenizer: Any, sequence_len: int) -> None:
        if isinstance(sequence_len, bool) or not isinstance(sequence_len, int) or sequence_len <= 0:
            raise ValueError(f"sequence_len must be a positive int, got {sequence_len!r}")
        self._tokenizer = tokenizer
        self._sequence_len = sequence_len
        self._tokens = [tokenizer.get_bos_token_id()]
        self._text = tokenizer.decode(self._tokens)
        self._marker_states = {
            tokenizer.encode_special("<user_start>"): TurnState.IN_USER,
            tokenizer.encode_special("<user_end>"): TurnState.BETWEEN,
            tokenizer.encode_special("<assistant_start>"): TurnState.IN_ASSISTANT,
            tokenizer.encode_special("<assistant_end>"): TurnState.BETWEEN,
        }

    def load_prefix(self, messages: list) -> None:
        if messages:
            rendered_ids, _ = self._tokenizer.render_conversation(
                {"messages": messages}, max_tokens=2**31
            )
            tokens = list(rendered_ids)
            assert_renderable(len(tokens), source="conversation")
        else:
            tokens = [self._tokenizer.get_bos_token_id()]

        text = self._tokenizer.decode(tokens)
        self._tokens = tokens
        self._text = text

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    @property
    def text(self) -> str:
        return self._text

    def turn_state(self) -> TurnState:
        return self.turn_state_at(len(self._tokens))

    def turn_state_at(self, token_count: int) -> TurnState:
        """Turn state of the first ``token_count`` committed tokens."""

        state = TurnState.BETWEEN
        for token_id in self._tokens[:token_count]:
            state = self._marker_states.get(token_id, state)
        return state

    def append_tokens(self, new_tokens: Sequence[int]) -> TextDelta:
        old_text = self._text
        self._tokens.extend(new_tokens)
        new_text = self._tokenizer.decode(self._tokens)
        self._text = new_text

        if new_text.startswith(old_text):
            return TextDelta(appended=new_text[len(old_text) :], rewritten_from=None)

        shared_prefix_len = 0
        for old_char, new_char in zip(old_text, new_text, strict=False):
            if old_char != new_char:
                break
            shared_prefix_len += 1
        return TextDelta(
            appended=new_text[shared_prefix_len:],
            rewritten_from=shared_prefix_len,
        )

    def truncate_to(self, n: int) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= len(self._tokens):
            raise ValueError(
                f"truncate token count must be an int in [0, {len(self._tokens)}], got {n!r}"
            )
        del self._tokens[n:]
        self._text = self._tokenizer.decode(self._tokens)

    def restore_tokens(self, tokens: tuple[int, ...]) -> None:
        """Reinstate a previously captured ``tokens`` snapshot verbatim."""

        self._tokens = list(tokens)
        self._text = self._tokenizer.decode(self._tokens)

    def char_offsets(self) -> list[int]:
        return token_to_char_offsets(self._tokenizer, self._tokens)

    def open_user_text(self) -> str | None:
        """Decoded text of the open partial user turn, or None when none is open."""

        if self.turn_state() is not TurnState.IN_USER:
            return None
        user_start = self._tokenizer.encode_special("<user_start>")
        last = max(index for index, token in enumerate(self._tokens) if token == user_start)
        return self._text[self.char_offsets()[last + 1] :]

    def forced_turn_tokens(self, text: str) -> list[int]:
        state = self.turn_state()
        if state is TurnState.IN_ASSISTANT:
            raise ValueError("assistant turn is still open")

        forced = list(self._tokenizer.encode(text))
        if state is TurnState.BETWEEN:
            forced.insert(0, self._tokenizer.encode_special("<user_start>"))
        forced.extend(
            [
                self._tokenizer.encode_special("<user_end>"),
                self._tokenizer.encode_special("<assistant_start>"),
            ]
        )
        return forced

    def preflight_user_turn(self, forced: Sequence[int]) -> None:
        required = len(self._tokens) + len(forced) + 1 + 1
        if required > self._sequence_len:
            raise ValueError(
                f"user turn requires {required} tokens total "
                f"({len(self._tokens)} committed + {len(forced)} forced + "
                "1 sampled token + 1 reserved <assistant_end> close), "
                f"but sequence_len is {self._sequence_len}"
            )

    def sampling_cap(self, max_new_tokens_per_turn: int) -> int:
        available = max(0, self._sequence_len - len(self._tokens) - 1)
        return max(0, min(max_new_tokens_per_turn, available))

    def can_commit(self, n: int) -> bool:
        return n >= 0 and len(self._tokens) + n <= self._sequence_len


def segment_char_spans(
    tokens: Sequence[int],
    char_offsets: Sequence[int],
    token_start: int,
    token_end: int,
) -> tuple[int, int]:
    # token_to_char_offsets returns len(tokens) + 1 boundaries, including the final text end.
    if len(char_offsets) != len(tokens) + 1:
        raise ValueError(
            "char_offsets must contain one boundary per token plus the final text boundary"
        )
    return char_offsets[token_start], char_offsets[token_end]


class ChatSession:
    """One continuous self-conditioned model stream for a chat conversation."""

    def __init__(
        self,
        *,
        setup,
        model,
        tokenizer,
        device,
        config: ChatSessionConfig,
        extraction: ExtractionService,
    ) -> None:
        self._setup = setup
        self._model = model
        self._tokenizer = tokenizer
        self._device = t.device(device)
        self._config = config
        self._extraction = extraction
        self._conversation = ConversationText(tokenizer, model.dec_config.sequence_len)
        self._start_messages: list[dict[str, Any]] = []

        self._token_rng = t.Generator(device=self._device)
        # seed + 1 deliberately keeps feature-count draws independent from token draws.
        self._selection_rng = t.Generator(device="cpu")
        self._attempt = 0
        self._reseed_generators()

        self._active_steering = SteeringSelection(SteeringMethod.OFF, ())
        head_width = (
            int(setup.feature_vocab.size)
            if setup.feature_vocab is not None
            else int(setup.n_sae_features)
        )
        self._score_policy = build_score_policy(
            self._active_steering,
            positive_selection=setup.output_selection,
            negative_selection=setup.output_negative_selection,
            head_width=head_width,
            feature_mask=setup.feature_mask,
            head_row_original_ids=(
                None if setup.feature_vocab is None else setup.feature_vocab.new_to_old_tensor()
            ),
        )

        self._wrapper: TargetAwareEncoderDecoderEvalWrapper | None = None
        self._kv_cache: KVCache | None = None
        self._ledger: list[LedgerSegment] = []
        self._snapshots: list[BoundarySnapshot] = []
        self._assistant_end_id = int(tokenizer.encode_special("<assistant_end>"))
        self._prefill_len: int | None = None
        self._aligned_segments: tuple[ContextFeatureSegment, ...] | None = None
        self._pending_s0_override: dict[int, int] | None = None
        self._pending_modify_retry: tuple[int, dict[int, int], tuple[int, ...]] | None = None
        self._cancel_requested = False
        # Set only while a FIRST prefill's turn tokens are provisionally appended;
        # consumed by the rollback handler (never cleared in _discard_runtime,
        # which the handler itself calls first).
        self._prefill_conversation_base: int | None = None

    def _reseed_generators(self) -> None:
        self._token_rng.manual_seed(int(self._config.seed))
        self._selection_rng.manual_seed(int(self._config.seed) + 1)

    def _discard_runtime(self) -> None:
        self._wrapper = None
        self._kv_cache = None
        self._ledger = []
        self._snapshots = []
        self._prefill_len = None
        self._aligned_segments = None
        self._pending_s0_override = None

    def _build_self_cond_config(self, policy) -> SelfCondConfig:
        fp_config = self._model.feature_pred_config
        if fp_config is None:
            raise ValueError("SelfConditioning requires checkpoint feature_prediction metadata")
        if FeaturePredLossMode(fp_config.loss_mode) is not FeaturePredLossMode.MAX_POOL:
            raise ValueError("SelfConditioning requires a MAX_POOL feature-prediction head")
        model_num_buckets = int(self._model.encoder.config.num_buckets)
        if model_num_buckets != 2:
            raise ValueError(
                "SelfConditioning chat sessions require two encoder buckets, "
                f"got {model_num_buckets}"
            )
        return build_session_self_cond_config(
            self._setup,
            self._config,
            self._selection_rng,
            policy,
            segment_mode=self._setup.segment_mode,
        )

    def _make_kv_cache(self) -> KVCache:
        cfg = self._model.dec_config
        dtype = t.bfloat16 if self._device.type == "cuda" else t.float32
        return KVCache(
            batch_size=1,
            num_heads=cfg.n_kv_head,
            seq_len=cfg.sequence_len,
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=self._model.num_cache_layers,
            device=self._device,
            dtype=dtype,
        )

    def load_prefix(self, messages: list) -> None:
        self._conversation.load_prefix(messages)
        self._start_messages = [dict(message) for message in messages]
        self._discard_runtime()
        self._attempt = 0
        self._reseed_generators()
        self._pending_modify_retry = None

    def clear(self) -> None:
        """Re-render the starting prefix and restart the complete RNG story."""
        self._conversation.load_prefix(self._start_messages)
        self._discard_runtime()
        self._attempt = 0
        self._reseed_generators()
        self._pending_modify_retry = None

    def set_seed(self, seed: int) -> None:
        """Replace the session seed; effective at the next RNG restart.

        The running conversation's generator streams are untouched — the new
        seed applies when ``clear``/``load_prefix`` reseed, or when a
        fresh-take rewind derives its attempt stream. This matches the UI's
        "seed N (next reset)" copy.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            raise ValueError(f"seed must be a positive int, got {seed!r}")
        self._config = replace(self._config, seed=seed)

    def regenerate(
        self,
        seed: int,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Replace the latest assistant response with a fresh seeded take.

        Earlier turns are replayed verbatim by ``resteer``. Failed operations
        retain the previous configured seed; deterministic rewind is unchanged.
        """
        original_config = self._config
        self.set_seed(seed)
        self._config = replace(self._config, deterministic_rewind=False)
        try:
            self.resteer(on_event, cancel_check=cancel_check)
        except Exception:
            self._config = original_config
            raise
        finally:
            self._config = replace(
                self._config, deterministic_rewind=original_config.deterministic_rewind
            )

    def set_steering(self, selection: SteeringSelection) -> None:
        head_width = (
            int(self._setup.feature_vocab.size)
            if self._setup.feature_vocab is not None
            else int(self._setup.n_sae_features)
        )
        policy = build_score_policy(
            selection,
            positive_selection=self._setup.output_selection,
            negative_selection=self._setup.output_negative_selection,
            head_width=head_width,
            feature_mask=self._setup.feature_mask,
            head_row_original_ids=(
                None
                if self._setup.feature_vocab is None
                else self._setup.feature_vocab.new_to_old_tensor()
            ),
        )
        if self._wrapper is not None:
            config = self._build_self_cond_config(policy)
            provider = SelfConditionedFeatureProvider(self._model, config, self._device)
            self._wrapper.swap_self_cond_provider(provider, config)
        self._active_steering = selection
        self._score_policy = policy
        self._pending_modify_retry = None

    def _restore_rng(
        self,
        snapshot: BoundarySnapshot | None,
        *,
        fresh_take: bool,
    ) -> None:
        if fresh_take and not self._config.deterministic_rewind:
            self._attempt += 1
            stride = 1_000_003
            self._token_rng.manual_seed(int(self._config.seed) + stride * self._attempt)
            self._selection_rng.manual_seed(int(self._config.seed) + 1 + stride * self._attempt)
            return
        if snapshot is None:
            self._reseed_generators()
            return
        self._token_rng.set_state(snapshot.token_rng_state)
        self._selection_rng.set_state(snapshot.selection_rng_state)

    def _restore_to_snapshot(self, ordinal: int, *, fresh_take: bool) -> t.Tensor:
        wrapper = self._wrapper
        kv_cache = self._kv_cache
        if wrapper is None or kv_cache is None:
            raise RuntimeError("cannot restore a snapshot before any generation")
        snapshot = self._snapshots[ordinal]
        kv_cache.rewind(int(kv_cache.get_pos()) - snapshot.kv_pos)
        wrapper.restore_self_cond_runtime(snapshot.runtime)
        self._restore_rng(snapshot, fresh_take=fresh_take)
        self._conversation.truncate_to(snapshot.token_count)
        del self._ledger[snapshot.ledger_len :]
        del self._snapshots[ordinal + 1 :]
        return snapshot.next_logits.to(self._device)

    def _emit_transcript(self, on_event: Callable[[GenerationEvent], None]) -> None:
        on_event(
            GenerationEvent(
                kind=GenerationEventKind.TRANSCRIPT,
                text_delta=None,
                detail=self._conversation.text,
            )
        )

    @contextmanager
    def _rollback_on_error(
        self,
        on_event: Callable[[GenerationEvent], None],
        *,
        entry_rng_states: tuple[t.Tensor, t.Tensor] | None = None,
    ) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            if self._wrapper is not None:
                self._runtime_provider().discard_explicit_segment()
            if self._snapshots and self._wrapper is not None and self._kv_cache is not None:
                snapshot = self._snapshots[-1]
                self._restore_to_snapshot(snapshot.selfcond_ordinal, fresh_take=False)
                detail = (
                    f"rolled back to boundary {snapshot.selfcond_ordinal} "
                    f"(kv {snapshot.kv_pos}): {exc}"
                )
            else:
                self._discard_runtime()
                # A first prefill failed: also roll back the turn tokens appended
                # for it, so a retry is an ordinary fresh send (any text).
                base_len = self._prefill_conversation_base
                if base_len is not None:
                    self._conversation.truncate_to(base_len)
                if entry_rng_states is not None:
                    token_state, selection_state = entry_rng_states
                    self._token_rng.set_state(token_state)
                    self._selection_rng.set_state(selection_state)
                detail = f"discarded prefill: {exc}"
            if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                detail += "; the assistant turn is open — rewind via /steer to continue"
            self._prefill_conversation_base = None
            on_event(
                GenerationEvent(
                    kind=GenerationEventKind.ERROR_ROLLBACK,
                    text_delta=None,
                    detail=detail,
                )
            )
            raise

    def _next_color(self) -> int:
        colors = [entry.color_idx for entry in self._ledger]
        left = colors[-1] if colors else None
        return next_color(colors, left_neighbor_idx=left)

    @staticmethod
    def _features(segment: ContextFeatureSegment) -> dict[int, int]:
        return {
            int(feature_id): int(bucket)
            for feature_id, bucket in zip(
                segment.feature_ids,
                segment.buckets,
                strict=True,
            )
        }

    def _steering_stamp(self) -> SteeringStamp | None:
        selection = self._active_steering
        if selection.method is SteeringMethod.OFF or not selection.enabled:
            return None
        total = math.fsum(strength for _signal, strength in selection.enabled)
        if selection.method is SteeringMethod.INTERPOLATION and total <= 0.0:
            return None
        return SteeringStamp(
            method=selection.method,
            signal_names=tuple(signal.name for signal, _strength in selection.enabled),
            total_strength=total,
        )

    def _append_prefix_ledger(
        self,
        prompt: tuple[int, ...],
        aligned: tuple[ContextFeatureSegment, ...],
        *,
        assignments: t.Tensor | None = None,
    ) -> None:
        if assignments is None:
            assignments = compute_context_token_segment_assignments(
                tokenizer=self._tokenizer,
                context_tokens=list(prompt),
                context_segments=aligned,
            )
        conditioned = assignments[:-1].tolist()
        positions_by_slot = [
            [
                position
                for position, assigned_slot in enumerate(conditioned)
                if int(assigned_slot) == slot_idx
            ]
            for slot_idx in range(len(aligned))
        ]
        empty_slots = [
            slot_idx for slot_idx, positions in enumerate(positions_by_slot) if not positions
        ]
        if empty_slots:
            final_slot_idx = len(aligned) - 1
            if len(empty_slots) == 1 and empty_slots[0] != final_slot_idx:
                raise RuntimeError(
                    "real prefix segment slot "
                    f"{empty_slots[0]} has no conditioned tokens before a later "
                    "non-empty slot"
                )
            if len(empty_slots) != 1:
                raise RuntimeError(
                    "real prefix ledger permits at most one empty slot, and only as "
                    "the final aligned slot"
                )
            if int(assignments[-1]) != final_slot_idx:
                raise RuntimeError(
                    "the final prompt token must route to the trailing empty real prefix slot"
                )
        for slot_idx, (segment, positions) in enumerate(
            zip(aligned, positions_by_slot, strict=True)
        ):
            if not positions:
                if any(positions_by_slot[later] for later in range(slot_idx + 1, len(aligned))):
                    raise RuntimeError(
                        "real prefix segment slot "
                        f"{slot_idx} has no conditioned tokens before a later non-empty slot"
                    )
                token_start = token_end = len(prompt) - 1
            else:
                token_start = positions[0]
                token_end = positions[-1] + 1
                if positions != list(range(token_start, token_end)):
                    raise RuntimeError(
                        f"real prefix segment slot {slot_idx} is not a contiguous token run"
                    )
            self._ledger.append(
                LedgerSegment(
                    kind=SegmentKind.PREFIX_REAL,
                    encoder_slot_idx=slot_idx,
                    selfcond_ordinal=None,
                    token_start=token_start,
                    token_end=token_end,
                    features=self._features(segment),
                    provenance=SegmentProvenance.REAL,
                    steering=None,
                    color_idx=self._next_color(),
                )
            )
        non_empty = [
            entry
            for entry in self._ledger
            if entry.kind is SegmentKind.PREFIX_REAL and entry.token_start != entry.token_end
        ]
        if (
            not non_empty
            or non_empty[0].token_start != 0
            or non_empty[-1].token_end != len(prompt) - 1
            or any(left.token_end != right.token_start for left, right in pairwise(non_empty))
        ):
            raise RuntimeError("real prefix ledger does not tile [0, prefill_len - 1)")

    def _runtime_provider(self):
        wrapper = self._wrapper
        if wrapper is None or wrapper._self_cond is None:
            raise RuntimeError("self-conditioned wrapper runtime is unavailable")
        return wrapper._self_cond.provider

    def _append_selfcond_ledger(
        self,
        *,
        ordinal: int,
        encoder_slot_idx: int,
        token_start: int,
        provenance_override: SegmentProvenance | None = None,
    ) -> None:
        features = self._runtime_provider().last_appended_segment
        if features is None:
            raise RuntimeError(
                f"self-conditioned provider did not record appended segment {ordinal}"
            )
        stamp = self._steering_stamp()
        self._ledger.append(
            LedgerSegment(
                kind=SegmentKind.SELFCOND,
                encoder_slot_idx=encoder_slot_idx,
                selfcond_ordinal=ordinal,
                token_start=token_start,
                token_end=token_start,
                features=dict(features),
                provenance=(
                    provenance_override
                    if provenance_override is not None
                    else (
                        SegmentProvenance.STEERED
                        if stamp is not None
                        else SegmentProvenance.PREDICTED
                    )
                ),
                steering=stamp,
                color_idx=self._next_color(),
            )
        )

    def _record_snapshot(self, ordinal: int, logits: t.Tensor) -> None:
        wrapper = self._wrapper
        kv_cache = self._kv_cache
        if wrapper is None or kv_cache is None:
            raise RuntimeError("cannot snapshot before wrapper prefill")
        runtime = wrapper.self_cond_runtime_snapshot()
        if runtime.fp_buffer != () or runtime.tokens_in_seg != 0:
            raise RuntimeError("boundary snapshot requires an empty self-cond window")
        token_count = len(self._conversation.tokens)
        kv_pos = int(kv_cache.get_pos())
        if kv_pos != token_count:
            raise RuntimeError(
                f"KV position {kv_pos} disagrees with conversation token count {token_count}"
            )
        self._snapshots.append(
            BoundarySnapshot(
                selfcond_ordinal=ordinal,
                kv_pos=kv_pos,
                runtime=runtime,
                next_logits=logits.detach().to(device="cpu", dtype=t.float32).clone(),
                token_rng_state=self._token_rng.get_state().clone(),
                selection_rng_state=self._selection_rng.get_state().clone(),
                token_count=token_count,
                ledger_len=len(self._ledger),
            )
        )

    def _extract_aligned(self, prompt: tuple[int, ...]) -> tuple[ContextFeatureSegment, ...]:
        prepared = self._extraction.prepare(prompt)
        if prepared.token_ids != prompt:
            raise ValueError("extraction prepared token_ids do not match the session prompt")
        if not prepared.segments:
            raise ValueError("extraction prepared no real prompt feature segments")
        if int(prepared.assistant_end_id) != self._assistant_end_id:
            raise ValueError(
                "extraction assistant_end_id "
                f"{int(prepared.assistant_end_id)} does not match tokenizer assistant_end_id "
                f"{self._assistant_end_id}"
            )
        return align_segments_to_decoded_prefix(
            tokenizer=self._tokenizer,
            prefix_tokens=list(prompt),
            segments=prepared.segments,
            counter_name=_CHAT_TUI_ALIGNMENT_COUNTER,
        )

    def _prefill(self, on_event: Callable[[GenerationEvent], None]) -> t.Tensor:
        prompt = self._conversation.tokens
        aligned = self._extract_aligned(prompt)
        return self._prefill_from_segments(prompt, aligned, on_event)

    def _build_prefill_world(
        self,
        prompt: tuple[int, ...],
        aligned: tuple[ContextFeatureSegment, ...],
    ) -> _PrefillWorld:
        """Build and forward a fresh wrapper/KV pair without touching session state.

        Failures here (extraction happened earlier, so chiefly the model forward)
        leave the session exactly as it was — callers that still hold a live world
        keep it. The pending S0 override is still popped before the attempt.
        """

        provider = SingleEntryContextProvider(prompt, aligned)
        config = self._build_self_cond_config(self._score_policy)
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=self._model,
            tokenizer=self._tokenizer,
            target=SelfConditionedTarget(),
            context_provider=provider,
            self_cond_config=config,
        )
        kv_cache = self._make_kv_cache()

        s0_override = self._pending_s0_override
        # Clear BEFORE arming: a failed arm must not poison later unrelated prefills.
        self._pending_s0_override = None
        if s0_override is not None:
            if wrapper._self_cond is None:
                raise RuntimeError("self-conditioned wrapper runtime is unavailable")
            wrapper._self_cond.provider.queue_explicit_segment(s0_override)

        input_ids = t.tensor([prompt], dtype=t.long, device=self._device)
        with wrapper.conditioning_context(context_token_ids=[list(prompt)]):
            logits = wrapper(input_ids, kv_cache=kv_cache)[:, -1, :]
        return _PrefillWorld(
            wrapper=wrapper,
            kv_cache=kv_cache,
            logits=logits,
            s0_overridden=s0_override is not None,
        )

    def _install_prefill_world(
        self,
        prompt: tuple[int, ...],
        aligned: tuple[ContextFeatureSegment, ...],
        world: _PrefillWorld,
        on_event: Callable[[GenerationEvent], None],
        *,
        overridden_slot: int | None = None,
    ) -> t.Tensor:
        self._wrapper = world.wrapper
        self._kv_cache = world.kv_cache
        self._prefill_len = len(prompt)
        self._aligned_segments = aligned

        self._append_prefix_ledger(prompt, aligned)
        if overridden_slot is not None:
            entry = self._ledger[overridden_slot]
            self._ledger[overridden_slot] = replace(
                entry,
                provenance=SegmentProvenance.OVERRIDDEN,
            )
        runtime = world.wrapper.self_cond_runtime_snapshot()
        self._append_selfcond_ledger(
            ordinal=0,
            encoder_slot_idx=runtime.state.target_slot_idx,
            token_start=len(prompt) - 1,
            provenance_override=(SegmentProvenance.OVERRIDDEN if world.s0_overridden else None),
        )
        self._record_snapshot(0, world.logits)
        on_event(
            GenerationEvent(
                kind=GenerationEventKind.SEGMENT_APPENDED,
                text_delta=None,
                detail="0",
            )
        )
        return world.logits

    def _prefill_from_segments(
        self,
        prompt: tuple[int, ...],
        aligned: tuple[ContextFeatureSegment, ...],
        on_event: Callable[[GenerationEvent], None],
        *,
        overridden_slot: int | None = None,
    ) -> t.Tensor:
        world = self._build_prefill_world(prompt, aligned)
        return self._install_prefill_world(
            prompt,
            aligned,
            world,
            on_event,
            overridden_slot=overridden_slot,
        )

    def _handle_boundary(
        self,
        logits: t.Tensor,
        on_event: Callable[[GenerationEvent], None],
        provenance_override: SegmentProvenance | None = None,
    ) -> None:
        wrapper = self._wrapper
        if wrapper is None:
            raise RuntimeError("cannot inspect a boundary without a wrapper")
        runtime = wrapper.self_cond_runtime_snapshot()
        if runtime.tokens_in_seg != 0:
            return
        if not self._ledger or self._ledger[-1].kind is not SegmentKind.SELFCOND:
            raise RuntimeError("self-conditioned boundary has no live ledger tail")
        previous = self._ledger[-1]
        if previous.selfcond_ordinal is None:
            raise RuntimeError("self-conditioned ledger tail has no ordinal")
        boundary_token = len(self._conversation.tokens) - 1
        self._ledger[-1] = replace(previous, token_end=boundary_token)
        ordinal = previous.selfcond_ordinal + 1
        self._append_selfcond_ledger(
            ordinal=ordinal,
            encoder_slot_idx=runtime.state.target_slot_idx,
            token_start=boundary_token,
            provenance_override=provenance_override,
        )
        self._record_snapshot(ordinal, logits)
        on_event(
            GenerationEvent(
                kind=GenerationEventKind.SEGMENT_APPENDED,
                text_delta=None,
                detail=str(ordinal),
            )
        )

    def _commit_token(
        self,
        token_id: int,
        on_event: Callable[[GenerationEvent], None],
        provenance_override: SegmentProvenance | None = None,
    ) -> tuple[t.Tensor, TextDelta]:
        if not self._conversation.can_commit(1):
            raise RuntimeError("cannot commit 1 token: decoder sequence budget is exhausted")
        wrapper = self._wrapper
        kv_cache = self._kv_cache
        if wrapper is None or kv_cache is None:
            raise RuntimeError("cannot decode before wrapper prefill")
        input_ids = t.tensor([[token_id]], dtype=t.long, device=self._device)
        logits = wrapper(input_ids, kv_cache=kv_cache)[:, -1, :]
        delta = self._conversation.append_tokens([token_id])
        self._handle_boundary(logits, on_event, provenance_override)
        if int(kv_cache.get_pos()) != len(self._conversation.tokens):
            raise RuntimeError("KV position disagrees with committed conversation tokens")
        return logits, delta

    def _sample_until_close(
        self,
        logits: t.Tensor,
        on_event: Callable[[GenerationEvent], None],
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        assistant_end_id = self._assistant_end_id
        bos_id = int(self._tokenizer.get_bos_token_id())
        cap = self._conversation.sampling_cap(self._config.max_new_tokens_per_turn)
        for _ in range(cap):
            cancelled = cancel_check() if cancel_check is not None else self._cancel_requested
            if cancelled:
                if cancel_check is None:
                    self._cancel_requested = False
                break
            next_id = sample_next_token(
                logits,
                self._token_rng,
                self._config.temperature,
                self._config.top_k,
            )
            token_id = int(next_id.item())
            if token_id in {assistant_end_id, bos_id}:
                break
            logits, delta = self._commit_token(token_id, on_event)
            on_event(
                GenerationEvent(
                    kind=GenerationEventKind.TOKEN,
                    text_delta=delta,
                    detail=str(token_id),
                )
            )

        _logits, close_delta = self._commit_token(assistant_end_id, on_event)
        on_event(
            GenerationEvent(
                kind=GenerationEventKind.TOKEN,
                text_delta=close_delta,
                detail=str(assistant_end_id),
            )
        )
        on_event(
            GenerationEvent(
                kind=GenerationEventKind.TURN_CLOSED,
                text_delta=None,
                detail="",
            )
        )

    def cancel(self) -> None:
        """Request cooperative cancellation of the active sampling loop."""

        self._cancel_requested = True

    @t.inference_mode()
    def send_user_turn(
        self,
        text: str,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._cancel_requested = False
        entry_rng_states = (
            self._token_rng.get_state().clone(),
            self._selection_rng.get_state().clone(),
        )
        forced = self._conversation.forced_turn_tokens(text)
        self._conversation.preflight_user_turn(forced)

        with self._rollback_on_error(on_event, entry_rng_states=entry_rng_states):
            if self._wrapper is None:
                self._prefill_conversation_base = len(self._conversation.tokens)
                self._conversation.append_tokens(forced)
                logits = self._prefill(on_event)
                self._prefill_conversation_base = None
                self._emit_transcript(on_event)
            else:
                for token_id in forced:
                    logits, delta = self._commit_token(int(token_id), on_event)
                    on_event(
                        GenerationEvent(
                            kind=GenerationEventKind.TOKEN,
                            text_delta=delta,
                            detail=str(int(token_id)),
                        )
                    )

            self._sample_until_close(logits, on_event, cancel_check)
        self._pending_modify_retry = None

    def _validate_snapshot_ordinal(self, ordinal: int) -> None:
        if self._wrapper is None:
            raise RuntimeError("cannot rewind or modify before any generation")
        if not 0 <= ordinal < len(self._snapshots):
            raise ValueError(
                f"snapshot ordinal {ordinal} is outside the valid range "
                f"0..{len(self._snapshots) - 1}"
            )

    @t.inference_mode()
    def rewind(
        self,
        ordinal: int,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._cancel_requested = False
        self._validate_snapshot_ordinal(ordinal)
        with self._rollback_on_error(on_event):
            logits = self._restore_to_snapshot(ordinal, fresh_take=True)
            self._emit_transcript(on_event)
            if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                self._sample_until_close(logits, on_event, cancel_check)
        self._pending_modify_retry = None

    @t.inference_mode()
    def _attempt_prefix_world(
        self,
        prompt: tuple[int, ...],
        aligned_source: Callable[[], tuple[ContextFeatureSegment, ...]],
        on_event: Callable[[GenerationEvent], None],
        failure_label: str,
    ) -> tuple[tuple[ContextFeatureSegment, ...], _PrefillWorld, _RngEntryState]:
        """Reseed and build a replacement world while the live one stays installed.

        The live wrapper/KV survive until the replacement has fully forwarded, so
        extraction, model-forward, and OOM failures abort the operation with the
        session exactly as it was (entry RNG states AND the fresh-take attempt
        counter restored — a transient failure must not change what a retry or
        any later fresh take generates). Both worlds are briefly resident; the
        decoder KV cache is small enough that this is fine.
        """

        entry_rng = _RngEntryState(
            token_state=self._token_rng.get_state().clone(),
            selection_state=self._selection_rng.get_state().clone(),
            attempt=self._attempt,
        )
        try:
            self._restore_rng(None, fresh_take=True)
            aligned = aligned_source()
            world = self._build_prefill_world(prompt, aligned)
        except Exception as exc:
            self._restore_rng_entry_state(entry_rng)
            on_event(
                GenerationEvent(
                    kind=GenerationEventKind.ERROR_ROLLBACK,
                    text_delta=None,
                    detail=f"{failure_label} failed before any state change: {exc}",
                )
            )
            raise
        return aligned, world, entry_rng

    def _restore_rng_entry_state(self, entry_rng: _RngEntryState) -> None:
        self._token_rng.set_state(entry_rng.token_state)
        self._selection_rng.set_state(entry_rng.selection_state)
        self._attempt = entry_rng.attempt

    def _swap_in_prefix_world(
        self,
        truncate_len: int,
        prompt: tuple[int, ...],
        aligned: tuple[ContextFeatureSegment, ...],
        world: _PrefillWorld,
        entry_rng: _RngEntryState,
        on_event: Callable[[GenerationEvent], None],
        failure_label: str,
        *,
        overridden_slot: int | None = None,
    ) -> t.Tensor:
        """Tear down the live world and install the built replacement atomically.

        Installation is pure bookkeeping (ledger, snapshot, events) but its
        internal invariant checks can raise; the live world's objects are all
        still referenced here, so any failure restores them (and the entry
        conversation tokens and RNG streams) wholesale instead of stranding an
        assistant-open session with no wrapper.
        """

        old_wrapper = self._wrapper
        old_kv_cache = self._kv_cache
        old_ledger = self._ledger
        old_snapshots = self._snapshots
        old_prefill_len = self._prefill_len
        old_aligned = self._aligned_segments
        old_pending_s0 = self._pending_s0_override
        old_tokens = self._conversation.tokens
        try:
            self._conversation.truncate_to(truncate_len)
            self._discard_runtime()
            return self._install_prefill_world(
                prompt,
                aligned,
                world,
                on_event,
                overridden_slot=overridden_slot,
            )
        except Exception as exc:
            self._wrapper = old_wrapper
            self._kv_cache = old_kv_cache
            self._ledger = old_ledger
            self._snapshots = old_snapshots
            self._prefill_len = old_prefill_len
            self._aligned_segments = old_aligned
            self._pending_s0_override = old_pending_s0
            self._conversation.restore_tokens(old_tokens)
            self._restore_rng_entry_state(entry_rng)
            on_event(
                GenerationEvent(
                    kind=GenerationEventKind.ERROR_ROLLBACK,
                    text_delta=None,
                    detail=(
                        f"{failure_label} failed during install; the previous "
                        f"session state was restored: {exc}"
                    ),
                )
            )
            raise

    @t.inference_mode()
    def rewind_to_prefix(
        self,
        ledger_index: int,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._cancel_requested = False
        if not 0 <= ledger_index < len(self._ledger):
            raise ValueError(f"ledger entry {ledger_index} is not a prefix segment")
        entry = self._ledger[ledger_index]
        if entry.kind is not SegmentKind.PREFIX_REAL:
            raise ValueError(f"ledger entry {ledger_index} is not a prefix segment")
        if entry.token_end < 2:
            raise ValueError("rewind_to_prefix requires at least two retained tokens")

        if self._conversation.turn_state_at(entry.token_end) is not TurnState.IN_ASSISTANT:
            # Lazy branch per spec: the next send prefills the exact retained
            # tokens together with the new turn, so no world is built here and
            # nothing after the (validated) truncation can fail.
            with self._rollback_on_error(on_event):
                self._conversation.truncate_to(entry.token_end)
                self._discard_runtime()
                self._restore_rng(None, fresh_take=True)
                self._emit_transcript(on_event)
            self._pending_modify_retry = None
            return

        prompt = self._conversation.tokens[: entry.token_end]
        aligned, world, entry_rng = self._attempt_prefix_world(
            prompt,
            lambda: self._extract_aligned(prompt),
            on_event,
            "prefix rewind",
        )
        logits = self._swap_in_prefix_world(
            entry.token_end,
            prompt,
            aligned,
            world,
            entry_rng,
            on_event,
            "prefix rewind",
        )
        self._emit_transcript(on_event)
        with self._rollback_on_error(on_event):
            self._sample_until_close(logits, on_event, cancel_check)
        self._pending_modify_retry = None

    @t.inference_mode()
    def modify_prefix(
        self,
        ledger_index: int,
        features: dict[int, int],
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Edit one real prefix slot and regenerate.

        Content after the original prefill prompt is discarded on success. The
        replacement world is fully built and forwarded BEFORE the live one is
        torn down, so a failed build leaves the session exactly as it was.
        Failed edits are not retry records.
        """

        self._cancel_requested = False
        if not 0 <= ledger_index < len(self._ledger):
            raise ValueError(f"ledger entry {ledger_index} is not a prefix segment")
        entry = self._ledger[ledger_index]
        if entry.kind is not SegmentKind.PREFIX_REAL:
            raise ValueError(f"ledger entry {ledger_index} is not a prefix segment")
        if self._wrapper is None:
            raise RuntimeError("cannot modify a prefix segment before any generation")
        validated = self._runtime_provider().validate_explicit_segment(features)
        original_aligned = self._aligned_segments
        prefill_len = self._prefill_len
        if original_aligned is None or prefill_len is None:
            raise RuntimeError("prefix alignment is unavailable")
        slot_idx = entry.encoder_slot_idx
        sorted_ids = tuple(sorted(validated))
        patched_segment = replace(
            original_aligned[slot_idx],
            feature_ids=sorted_ids,
            buckets=tuple(validated[feature_id] for feature_id in sorted_ids),
        )
        patched = (
            *original_aligned[:slot_idx],
            patched_segment,
            *original_aligned[slot_idx + 1 :],
        )
        prompt = self._conversation.tokens[:prefill_len]

        _aligned, world, entry_rng = self._attempt_prefix_world(
            prompt,
            lambda: patched,
            on_event,
            "prefix edit",
        )
        logits = self._swap_in_prefix_world(
            prefill_len,
            prompt,
            patched,
            world,
            entry_rng,
            on_event,
            "prefix edit",
            overridden_slot=slot_idx,
        )
        self._emit_transcript(on_event)
        if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
            with self._rollback_on_error(on_event):
                self._sample_until_close(logits, on_event, cancel_check)
        self._pending_modify_retry = None

    @t.inference_mode()
    def modify(
        self,
        ordinal: int,
        features: dict[int, int],
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._cancel_requested = False
        retry = self._pending_modify_retry
        matching_retry = retry is not None and retry[0] == ordinal
        retrying = matching_retry and ordinal == 0 and self._wrapper is None
        if matching_retry and ordinal >= 1:
            if self._wrapper is None or len(self._snapshots) != ordinal:
                raise ValueError(
                    f"snapshot ordinal {ordinal} is outside the valid range "
                    f"0..{len(self._snapshots) - 1}"
                )
            retrying = True
        if not retrying:
            self._validate_snapshot_ordinal(ordinal)
        if not features:
            raise ValueError("modify requires at least one feature")
        # Pre-validate against the LIVE provider before any teardown/replay:
        # an invalid edit must never cost conversation state.
        validation_provider = (
            self._runtime_provider()
            if self._wrapper is not None
            else SelfConditionedFeatureProvider(
                self._model,
                self._build_self_cond_config(self._score_policy),
                self._device,
            )
        )
        validated = validation_provider.validate_explicit_segment(features)
        features = validated
        if retrying:
            self._pending_modify_retry = None

        if ordinal == 0:
            token_count = (
                len(self._conversation.tokens) if retrying else self._snapshots[0].token_count
            )
            entry_rng_states = (
                self._token_rng.get_state().clone(),
                self._selection_rng.get_state().clone(),
            )
            override_landed = False
            try:
                with self._rollback_on_error(on_event, entry_rng_states=entry_rng_states):
                    self._conversation.truncate_to(token_count)
                    self._discard_runtime()
                    self._restore_rng(None, fresh_take=True)
                    self._emit_transcript(on_event)
                    self._pending_s0_override = dict(features)
                    logits = self._prefill(on_event)
                    override_landed = True
                    edited = self._ledger[-1]
                    if (
                        edited.selfcond_ordinal != 0
                        or edited.features != features
                        or edited.provenance is not SegmentProvenance.OVERRIDDEN
                    ):
                        raise RuntimeError("S0 override did not produce the requested segment")
                    if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                        self._sample_until_close(logits, on_event, cancel_check)
            except Exception:
                if not override_landed:
                    self._pending_modify_retry = (0, dict(features), ())
                raise
            self._pending_modify_retry = None
            return

        previous_ordinal = ordinal - 1
        previous = self._snapshots[previous_ordinal]
        segment_size = int(self._config.self_cond_segment_size)
        replay_tokens = (
            retry[2]
            if retrying and retry is not None
            else self._conversation.tokens[
                previous.token_count : previous.token_count + segment_size
            ]
        )
        if len(replay_tokens) != segment_size:
            raise RuntimeError(
                f"modify replay for ordinal {ordinal} requires {segment_size} recorded "
                f"tokens, got {len(replay_tokens)}"
            )

        override_landed = False
        try:
            with self._rollback_on_error(on_event):
                logits = self._restore_to_snapshot(previous_ordinal, fresh_take=True)
                self._emit_transcript(on_event)
                for token_id in replay_tokens[:-1]:
                    logits, delta = self._commit_token(int(token_id), on_event)
                    on_event(
                        GenerationEvent(
                            kind=GenerationEventKind.TOKEN,
                            text_delta=delta,
                            detail=str(int(token_id)),
                        )
                    )

                self._runtime_provider().queue_explicit_segment(features)
                logits, delta = self._commit_token(
                    int(replay_tokens[-1]),
                    on_event,
                    SegmentProvenance.OVERRIDDEN,
                )
                override_landed = True
                on_event(
                    GenerationEvent(
                        kind=GenerationEventKind.TOKEN,
                        text_delta=delta,
                        detail=str(int(replay_tokens[-1])),
                    )
                )
                edited = self._ledger[-1]
                if (
                    edited.selfcond_ordinal != ordinal
                    or edited.features != features
                    or edited.provenance is not SegmentProvenance.OVERRIDDEN
                ):
                    raise RuntimeError(
                        f"override for ordinal {ordinal} did not produce the requested segment"
                    )
                if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                    self._sample_until_close(logits, on_event, cancel_check)
        except Exception:
            if not override_landed:
                self._pending_modify_retry = (ordinal, dict(features), tuple(replay_tokens))
            raise
        self._pending_modify_retry = None

    @t.inference_mode()
    def resteer(
        self,
        on_event: Callable[[GenerationEvent], None],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Regenerate the current assistant turn under the CURRENT policy.

        The target is the segment that governs the live response — the one
        containing its ``<assistant_start>`` — so a steering change rewrites the
        answer the user is looking at. Targeting the most recent boundary
        instead would append a fresh continuation to a turn that had already
        closed and leave the visible answer unsteered.

        The recorded span into the governing boundary is replayed verbatim
        (independent of the deterministic-rewind setting), so the re-appended
        boundary samples its features with the just-applied steering. All
        recorded content through the FINAL ``<assistant_start>`` is preserved
        exactly — user-authored text is never regenerable — and only the final
        assistant response resamples under the new conditioning. With only the
        prefill boundary crossed, the prompt is re-prefilled atomically with
        its existing alignment instead (no re-extraction). A mid-replay
        failure rolls back to the previous boundary.
        """

        self._cancel_requested = False
        if self._wrapper is None:
            raise RuntimeError("cannot resteer before any generation")
        if self._conversation.turn_state() is TurnState.IN_USER:
            raise ValueError("cannot resteer over an open partial user turn")

        governing = self._governing_ordinal()
        if governing >= 1:
            previous = self._snapshots[governing - 1]
            segment_size = int(self._config.self_cond_segment_size)
            boundary_count = previous.token_count + segment_size
            replay_tokens = self._conversation.tokens[previous.token_count : boundary_count]
            if len(replay_tokens) != segment_size:
                raise RuntimeError(
                    f"resteer replay for ordinal {governing} requires {segment_size} "
                    f"recorded tokens, got {len(replay_tokens)}"
                )
            recorded_tail = self._conversation.tokens[boundary_count:]
            forced_tail = recorded_tail[: self._forced_tail_count(recorded_tail)]
            with self._rollback_on_error(on_event):
                logits = self._restore_to_snapshot(governing - 1, fresh_take=True)
                self._emit_transcript(on_event)
                for token_id in (*replay_tokens, *forced_tail):
                    logits, delta = self._commit_token(int(token_id), on_event)
                    on_event(
                        GenerationEvent(
                            kind=GenerationEventKind.TOKEN,
                            text_delta=delta,
                            detail=str(int(token_id)),
                        )
                    )
                if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                    self._sample_until_close(logits, on_event, cancel_check)
            self._pending_modify_retry = None
            return

        prefill_len = self._prefill_len
        aligned = self._aligned_segments
        if prefill_len is None or aligned is None:
            raise RuntimeError("prefix alignment is unavailable")
        prompt = self._conversation.tokens[:prefill_len]
        recorded_tail = self._conversation.tokens[prefill_len:]
        forced_tail = recorded_tail[: self._forced_tail_count(recorded_tail)]
        _aligned, world, entry_rng = self._attempt_prefix_world(
            prompt,
            lambda: aligned,
            on_event,
            "steering reselect",
        )
        logits = self._swap_in_prefix_world(
            prefill_len,
            prompt,
            aligned,
            world,
            entry_rng,
            on_event,
            "steering reselect",
        )
        self._emit_transcript(on_event)
        with self._rollback_on_error(on_event):
            for token_id in forced_tail:
                logits, delta = self._commit_token(int(token_id), on_event)
                on_event(
                    GenerationEvent(
                        kind=GenerationEventKind.TOKEN,
                        text_delta=delta,
                        detail=str(int(token_id)),
                    )
                )
            if self._conversation.turn_state() is TurnState.IN_ASSISTANT:
                self._sample_until_close(logits, on_event, cancel_check)
        self._pending_modify_retry = None

    def _governing_ordinal(self) -> int:
        """Ordinal of the segment whose conditioning governs the live turn.

        ``resteer`` must re-draw the conditioning the CURRENT assistant response
        was generated under, which is the segment containing that response's
        ``<assistant_start>`` — not simply the most recent boundary. When a turn
        has already closed, the newest segment covers only the tokens after it
        and re-crossing that boundary would append a fresh continuation while
        leaving the visible answer untouched.
        """

        tokens = self._conversation.tokens
        assistant_start = int(self._tokenizer.encode_special("<assistant_start>"))
        turn_start = next(
            (
                index + 1
                for index in range(len(tokens) - 1, -1, -1)
                if int(tokens[index]) == assistant_start
            ),
            None,
        )
        if turn_start is None:
            raise ValueError("cannot resteer before an assistant turn has begun")
        return max(
            ordinal
            for ordinal, snapshot in enumerate(self._snapshots)
            if snapshot.token_count <= turn_start
        )

    def _forced_tail_count(self, recorded_tail: tuple[int, ...]) -> int:
        """Tokens of a recorded tail to force verbatim during ``resteer``.

        Everything through the FINAL ``<assistant_start>`` is forced (earlier
        exchanges and user-authored text are preserved exactly); only the
        recorded text after it — the final assistant response — is discarded
        for regeneration.
        """

        assistant_start = int(self._tokenizer.encode_special("<assistant_start>"))
        for index in range(len(recorded_tail) - 1, -1, -1):
            if int(recorded_tail[index]) == assistant_start:
                return index + 1
        return 0

    def discard_preview(
        self,
        entry: LedgerSegment,
        *,
        is_modify: bool = False,
    ) -> DiscardPreview:
        """Preview the exact truncation boundary used by a rewind or modify."""

        operation = "modify" if is_modify else "rewind"
        if entry.kind is SegmentKind.PREFIX_REAL:
            retained = self._prefill_len if is_modify else entry.token_end
            if retained is None:
                raise RuntimeError("cannot preview a prefix operation before prefill")
        else:
            ordinal = entry.selfcond_ordinal
            if ordinal is None:
                raise RuntimeError("self-conditioned ledger segment has no ordinal")
            self._validate_snapshot_ordinal(ordinal)
            if not is_modify:
                retained = self._snapshots[ordinal].token_count
            elif ordinal == 0:
                retained = self._snapshots[0].token_count
            else:
                retained = self._snapshots[ordinal - 1].token_count + int(
                    self._config.self_cond_segment_size
                )
        return DiscardPreview(
            retained_tokens=retained,
            discarded_tokens=max(0, len(self._conversation.tokens) - retained),
            discarded_segments=sum(
                ledger_entry.token_start >= retained for ledger_entry in self._ledger
            ),
            operation=operation,
        )

    def ledger(self) -> tuple[LedgerSegment, ...]:
        if not self._ledger:
            return ()
        entries = list(self._ledger)
        live = entries[-1]
        if live.kind is SegmentKind.SELFCOND:
            entries[-1] = replace(
                live,
                token_end=len(self._conversation.tokens),
            )
        return tuple(entries)

    def token_budget(self) -> tuple[int, int]:
        return len(self._conversation.tokens), int(self._model.dec_config.sequence_len)

    def conversation_tokens(self) -> tuple[int, ...]:
        return self._conversation.tokens

    def open_user_text(self) -> str | None:
        """Decoded text of the open partial user turn, or None when none is open."""

        return self._conversation.open_user_text()

    def transcript_spans(self) -> tuple[str, tuple[TranscriptSpan, ...]]:
        """Return one authoritative decode plus ledger styles at character offsets."""

        text = self._conversation.text
        ledger = self.ledger()
        if not ledger:
            return text, ()
        tokens = self._conversation.tokens
        offsets = self._conversation.char_offsets()
        token_cursor = 0
        char_cursor = 0
        last_index = len(ledger) - 1
        spans: list[TranscriptSpan] = []
        for index, entry in enumerate(ledger):
            if entry.token_start != token_cursor:
                raise RuntimeError("ledger token spans must tile the conversation")
            char_start, char_end = segment_char_spans(
                tokens,
                offsets,
                entry.token_start,
                entry.token_end,
            )
            if char_start != char_cursor:
                raise RuntimeError("ledger character spans must tile the transcript")
            spans.append(
                TranscriptSpan(
                    char_start=char_start,
                    char_end=char_end,
                    color_idx=entry.color_idx,
                    kind=entry.kind,
                    is_live=index == last_index and entry.kind is SegmentKind.SELFCOND,
                )
            )
            token_cursor = entry.token_end
            char_cursor = char_end
        if token_cursor != len(tokens) or char_cursor != len(text):
            raise RuntimeError("ledger spans do not cover the authoritative transcript")
        return text, tuple(spans)

    def turn_state(self) -> TurnState:
        return self._conversation.turn_state()
