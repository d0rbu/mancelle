"""Bounded group-local reuse of frozen canonical feature-head inputs.

Sessions never modify model registries, global callables or rollout objects.
Complete normalized head inputs, fresh prompt snapshots and optional canonical
prompt/history encoder outputs are retained; the live head, validation, likelihood and backward
remain policy-owned. Frozen
model .data writes, unregistered independent-version aliases and concurrent
mutation are outside the frozen-trunk contract. Prompt contents are compared
afresh, including .data and alias writes. Optional allocation admission estimates
driver-free plus reusable reserve; only cache-copy CUDA OOM is suppressed.
"""

from __future__ import annotations

import json
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from threading import RLock, get_ident
from typing import Any

import torch as t

from steering.conditional.encoder_decoder import packed_encoder as packed
from steering.conditional.encoder_decoder._rlvr_replay_guard import NativeBinding, NativeFrozenState
from steering.conditional.encoder_decoder.sft_grpo import ConditioningAction

DEFAULT_MIN_FREE_BYTES = 4 * 1024**3
MAX_ENTRIES_PER_GROUP = 512
MAX_KEY_ELEMENTS_PER_GROUP = 262144
MAX_KEY_BYTES_PER_GROUP = 8 * 1024**2
MAX_REASON_KEYS = 32
MAX_PREFIX_KEYS = 64
_PROMPT_FIELDS = ("enc_feature_ids", "enc_magnitudes", "enc_mask", "prefix_seg_map")
_PROMPT_DTYPES = (t.int64, t.int64, t.bool, t.int64)
# Sessions hold their model strongly until close, so its identity cannot be
# recycled while claimed. This registry stores only owner tokens, never models.
_SESSION_OWNERS: dict[int, object] = {}
_SESSION_LOCK = RLock()


@dataclass(frozen=True)
class HiddenReplaySettings:
    max_cache_bytes: int = 0
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    history_encoder: bool = False
    prompt_encoder: bool = False

    def __post_init__(self) -> None:
        for name in ("max_cache_bytes", "min_free_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"hidden_replay_{name} must be a nonnegative integer")
        if type(self.history_encoder) is not bool:
            raise ValueError("hidden_replay_history_encoder must be a boolean")
        if type(self.prompt_encoder) is not bool:
            raise ValueError("hidden_replay_prompt_encoder must be a boolean")
        if self.prompt_encoder and not self.history_encoder:
            raise ValueError("hidden_replay_prompt_encoder requires hidden_replay_history_encoder")


def hidden_replay_settings(cfg: Mapping[str, Any]) -> HiddenReplaySettings:
    """Resolve optional execution controls without model or framework inspection."""
    return HiddenReplaySettings(
        max_cache_bytes=cfg.get("hidden_replay_max_cache_bytes", 0),
        min_free_bytes=cfg.get("hidden_replay_min_free_bytes", DEFAULT_MIN_FREE_BYTES),
        history_encoder=cfg.get("hidden_replay_history_encoder", False),
        prompt_encoder=cfg.get("hidden_replay_prompt_encoder", False),
    )


@dataclass
class HiddenReplayReport:
    """One bounded worker aggregate; never owns sessions, tensors or exceptions."""

    applied: bool = False
    cleaned_up: bool = True
    requested_groups: int = 0
    inactive_groups: int = 0
    inactive_reasons: dict[str, int] = field(default_factory=dict)
    active_groups: int = 0
    admitted_groups: int = 0
    groups_collected: int = 0
    groups_replayed: int = 0
    groups_completed: int = 0
    groups_closed: int = 0
    groups_abandoned: int = 0
    groups_failed: int = 0
    behavior_calls: int = 0
    capture_attempts: int = 0
    admitted_entries: int = 0
    capture_skips: int = 0
    replay_calls: int = 0
    replay_hits: int = 0
    replay_misses: int = 0
    invalidations: int = 0
    invalidation_reasons: dict[str, int] = field(default_factory=dict)
    miss_reasons: dict[str, int] = field(default_factory=dict)
    capture_skip_reasons: dict[str, int] = field(default_factory=dict)
    current_bytes: int = 0
    current_hidden_bytes: int = 0
    current_prompt_bytes: int = 0
    peak_bytes: int = 0
    peak_hidden_bytes: int = 0
    total_captured_bytes: int = 0
    # Existing peaks describe head entries; history_peak_entries is separate.
    # Admission nevertheless applies all three caps to their combined totals.
    peak_entries: int = 0
    peak_key_elements: int = 0
    peak_key_bytes: int = 0
    max_entries_per_group: int = MAX_ENTRIES_PER_GROUP
    max_key_elements_per_group: int = MAX_KEY_ELEMENTS_PER_GROUP
    max_key_bytes_per_group: int = MAX_KEY_BYTES_PER_GROUP
    prefix_length_counts: dict[str, int] = field(default_factory=dict)
    guard_calls: int = 0
    guard_seconds: float = 0.0
    capture_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    optional_copy_ooms: int = 0
    owned_tensor_objects_alive_after_cleanup: int = 0
    max_owned_tensor_objects_alive_after_cleanup: int = 0
    admission_free_bytes: int = 0
    admission_reusable_reserved_bytes: int = 0
    admission_estimated_available_bytes: int = 0
    history_calls: int = 0
    history_guard_calls: int = 0
    history_action_calls: int = 0
    history_hits: int = 0
    history_misses: int = 0
    history_admitted_entries: int = 0
    history_admission_skips: int = 0
    history_skip_reasons: dict[str, int] = field(default_factory=dict)
    history_evictions: int = 0
    history_clone_ooms: int = 0
    current_history_bytes: int = 0
    peak_history_bytes: int = 0
    history_peak_entries: int = 0
    history_total_captured_bytes: int = 0
    canonical_prompt_calls: int = 0
    canonical_prompt_hits: int = 0
    canonical_prompt_misses: int = 0
    canonical_prompt_admitted_entries: int = 0
    canonical_prompt_admission_skips: int = 0
    canonical_prompt_skip_reasons: dict[str, int] = field(default_factory=dict)
    canonical_prompt_evictions: int = 0
    canonical_prompt_clone_ooms: int = 0
    current_canonical_prompt_bytes: int = 0
    peak_canonical_prompt_bytes: int = 0
    canonical_prompt_total_captured_bytes: int = 0
    canonical_prompt_peak_entries: int = 0
    canonical_prompt_peak_key_elements: int = 0
    canonical_prompt_peak_key_bytes: int = 0


def hidden_replay_report_snapshot(report: HiddenReplayReport) -> dict[str, Any]:
    """Copy plain counters without synchronization or retaining group state."""
    return asdict(report)


def _bump(counts: dict[str, int], reason: str, *, limit: int = MAX_REASON_KEYS) -> None:
    reason = reason[:256]
    key = reason if reason in counts or len(counts) < limit - 1 else "other"
    counts[key] = counts.get(key, 0) + 1


def _clone_for_cache(value: t.Tensor) -> t.Tensor:
    """The sole optional allocation seam whose CUDA OOM may be suppressed."""
    with t.inference_mode(False), t.no_grad():
        return value.detach().clone(memory_format=t.preserve_format)


def _tensor_meta(value: t.Tensor) -> tuple[Any, ...]:
    return (value.device, value.dtype, value.layout, tuple(value.shape), tuple(value.stride()), value.storage_offset())


def _bytes(value: t.Tensor) -> int:
    return value.untyped_storage().nbytes()


def _ordinary_ints(value: Any, limit: int) -> bool:
    return type(value) in (tuple, list) and len(value) <= limit and all(
        type(item) is int and item.bit_length() <= 64 for item in value
    )


def _key(prompt: Any, prefix: Any, routing: Any, positions: Any,
         history: Any) -> tuple[tuple[Any, ...] | None, int, int, str]:
    limit = MAX_KEY_ELEMENTS_PER_GROUP
    if type(prompt.encoder_slot_count) is not int or prompt.encoder_slot_count.bit_length() > 64:
        return None, 0, 0, "unsupported_prompt"
    sequences = (prompt.prompt_token_ids, prefix, routing, positions)
    if not all(_ordinary_ints(value, limit) for value in sequences):
        return None, 0, 0, "unsupported_key_sequence"
    if type(history) not in (tuple, list) or len(history) > MAX_ENTRIES_PER_GROUP:
        return None, 0, 0, "unsupported_history"
    elements = sum(len(value) for value in sequences) + 1
    for action in history:
        if (type(action) is not ConditioningAction or type(action.feature_ids) is not tuple
                or type(action.magnitudes) is not tuple or type(action.mask) is not tuple
                or not _ordinary_ints(action.feature_ids, limit) or not _ordinary_ints(action.magnitudes, limit)
                or len(action.mask) > limit or any(type(item) is not bool for item in action.mask)):
            return None, 0, 0, "unsupported_history_action"
        elements += len(action.feature_ids) + len(action.magnitudes) + len(action.mask)
        if elements > limit:
            return None, 0, 0, "key_elements_limit"
    if elements > limit:
        return None, 0, 0, "key_elements_limit"
    key = (tuple(prompt.prompt_token_ids), prompt.encoder_slot_count, tuple(prefix), tuple(routing), tuple(positions),
           tuple((action.feature_ids, action.magnitudes, action.mask) for action in history))
    encoded_bytes = len(json.dumps(key, separators=(",", ":"), allow_nan=False).encode())
    if encoded_bytes > MAX_KEY_BYTES_PER_GROUP:
        return None, 0, 0, "key_bytes_limit"
    return key, elements, encoded_bytes, "eligible"


def _prompt_tensors(prompt: Any) -> tuple[t.Tensor, ...] | None:
    values = tuple(getattr(prompt, name) for name in _PROMPT_FIELDS)
    if not all(type(value) is t.Tensor and value.dtype == dtype and value.layout == t.strided
               and not value.requires_grad and value.device.type in ("cpu", "cuda")
               for value, dtype in zip(values, _PROMPT_DTYPES, strict=True)):
        return None
    return values


def _history_key(action: Any) -> tuple[tuple[Any, ...], int, int] | None:
    limit = MAX_KEY_ELEMENTS_PER_GROUP
    if (type(action) is not ConditioningAction or type(action.feature_ids) is not tuple
            or type(action.magnitudes) is not tuple or type(action.mask) is not tuple
            or not _ordinary_ints(action.feature_ids, limit) or not _ordinary_ints(action.magnitudes, limit)
            or len(action.mask) > limit or any(type(item) is not bool for item in action.mask)):
        return None
    key = (action.feature_ids, action.magnitudes, action.mask)
    elements = sum(len(value) for value in key)
    if elements > limit:
        return None
    key_bytes = len(json.dumps(key, separators=(",", ":")).encode())
    return (key, elements, key_bytes) if key_bytes <= MAX_KEY_BYTES_PER_GROUP else None


def _history_keys(history: Sequence[ConditioningAction]) -> list[tuple[tuple[Any, ...], int, int]] | None:
    keys = []
    total_elements = total_key_bytes = 0
    for action in history:
        parts = _history_key(action)
        if parts is None:
            return None
        total_elements += parts[1]
        total_key_bytes += parts[2]
        if total_elements > MAX_KEY_ELEMENTS_PER_GROUP or total_key_bytes > MAX_KEY_BYTES_PER_GROUP:
            return None
        keys.append(parts)
    return keys


def _packed_tensors(output: packed.PackedEncoderOutput) -> tuple[t.Tensor, ...]:
    layout = output.layout
    return (output.values, layout.flat_indices, layout.valid_mask, layout.batch_indices, layout.segment_indices)


@dataclass
class _HistoryEntry:
    output: packed.PackedEncoderOutput
    key_elements: int
    key_bytes: int


@dataclass
class _CanonicalPromptEntry:
    inputs: tuple[t.Tensor, ...]
    input_metadata: tuple[tuple[Any, ...], ...]
    output: packed.PackedEncoderOutput
    key_elements: int
    key_bytes: int

    def matches(self, inputs: tuple[t.Tensor, ...]) -> bool:
        return tuple(_tensor_meta(value) for value in inputs) == self.input_metadata and all(
            t.equal(actual, saved) for actual, saved in zip(inputs, self.inputs, strict=True)
        )


@dataclass(frozen=True)
class CanonicalReplayRequest:
    key: tuple[Any, ...]
    key_elements: int
    key_bytes: int
    prompt_tensors: tuple[t.Tensor, ...]
    prompt_metadata: tuple[tuple[Any, ...], ...]
    prefix_length: int
    owner: object


@dataclass(frozen=True)
class CachedHeadInput:
    hidden: t.Tensor
    encoder_slots: int


@dataclass
class _Entry:
    hidden: t.Tensor
    prompt_tensors: tuple[t.Tensor, ...]
    prompt_metadata: tuple[tuple[Any, ...], ...]
    encoder_slots: int

    def matches(self, request: CanonicalReplayRequest) -> bool:
        return request.prompt_metadata == self.prompt_metadata and all(
            _tensor_meta(actual) == metadata and t.equal(actual, saved)
            for actual, saved, metadata in zip(
                request.prompt_tensors, self.prompt_tensors, self.prompt_metadata, strict=True
            )
        )


class RLVRReplaySession:
    """One serialized model/group lifetime, created by the policy-owned factory."""

    def __init__(self, model: Any, *, settings: HiddenReplaySettings, report: HiddenReplayReport,
                 prompt_type: type, group_type: type, policy_dependencies: tuple[NativeBinding, ...]):
        self.settings, self.report = settings, report
        self._model, self._group = model, None
        self._prompt_type, self._group_type = prompt_type, group_type
        self._dependencies = policy_dependencies
        self._phase, self._thread = "ready", get_ident()
        self._owner = object()
        self._model_identity = id(model)
        self._guard: NativeFrozenState | None = None
        self._invalid_reason: str | None = None
        self._entries: dict[tuple[Any, ...], list[_Entry]] = {}
        self._history_entries: dict[tuple[Any, ...], _HistoryEntry] = {}
        self._history_pending: dict[tuple[Any, ...], _HistoryEntry] = {}
        self._canonical_prompt_entry: _CanonicalPromptEntry | None = None
        self._canonical_prompt_pending: _CanonicalPromptEntry | None = None
        self._canonical_prompt_committed_once = False
        self._history_key_elements = self._history_key_bytes = 0
        # These exact definition-time bindings are checked by the same frozen
        # guard before use; callers cannot supply replacement compute callbacks.
        self._action_tensors = next(binding.expected for binding in policy_dependencies
                                    if binding.name == "_action_tensors")
        self._append_history = next(binding.expected for binding in policy_dependencies
                                    if binding.name == "append_packed_encoder_segment")
        self._entry_count = self._key_elements = self._key_bytes = 0
        self._refs: list[weakref.ReferenceType[t.Tensor]] = []
        self._admitted = self._failed = self._completed = False
        with _SESSION_LOCK:
            if report.active_groups:
                raise RuntimeError("Hidden replay groups sharing a report must not overlap")
            if self._model_identity in _SESSION_OWNERS:
                raise RuntimeError("Hidden replay groups sharing a model must not overlap")
            _SESSION_OWNERS[self._model_identity] = self._owner
            report.active_groups += 1
            report.cleaned_up = False

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("Hidden replay session requires its owning thread")

    def begin_collection(self, model: Any) -> bool:
        self._check_thread()
        if self._phase != "ready":
            raise RuntimeError("Hidden replay collection requires a fresh group session")
        if model is not self._model:
            return False
        self._phase = "collecting"
        return True

    def seal_collection(self, model: Any, group: object) -> None:
        self._check_thread()
        if self._phase != "collecting" or model is not self._model or type(group) is not self._group_type:
            raise RuntimeError("Hidden replay collection must seal its exact native model/group")
        self._group, self._phase = group, "sealed"
        self._clear_history()
        self.report.groups_collected += 1

    @contextmanager
    def replaying(self, model: Any, group: object, *, train_feature_head_only: bool) -> Iterator[RLVRReplaySession | None]:
        self._check_thread()
        if self._phase != "sealed":
            raise RuntimeError("Hidden replay requires one sealed, unreplayed group")
        matches = model is self._model and group is self._group and train_feature_head_only is True
        if matches:
            self._phase = "replaying"
            self.report.groups_replayed += 1
        else:
            self._clear_entries()
        try:
            yield self if matches else None
            if matches:
                self._completed = True
                self.report.groups_completed += 1
        except BaseException:
            self.mark_failed()
            raise
        finally:
            self.close()

    def mark_failed(self) -> None:
        if not self._failed and self._phase != "closed":
            self._failed = True
            self.report.groups_failed += 1

    @property
    def failed(self) -> bool:
        """A native group error may have been handled inside the outer context."""
        return self._failed

    def _clear_entries(self) -> None:
        self._clear_history()
        self._entries.clear()
        self._entry_count = self._key_elements = self._key_bytes = 0
        self.report.current_bytes = self.report.current_hidden_bytes = self.report.current_prompt_bytes = 0

    def _release_history_output(self, output: packed.PackedEncoderOutput) -> None:
        size = sum(_bytes(value) for value in _packed_tensors(output))
        self.report.current_history_bytes -= size
        self.report.current_bytes -= size

    def discard_pending_history(self) -> None:
        """Discard unvalidated history and prompt outputs after one canonical call."""
        self._check_thread()
        if self._canonical_prompt_pending is not None:
            self._release_canonical_prompt_entry(self._canonical_prompt_pending)
            self._canonical_prompt_pending = None
        for entry in self._history_pending.values():
            self._release_history_output(entry.output)
            self._history_key_elements -= entry.key_elements
            self._history_key_bytes -= entry.key_bytes
        self._history_pending.clear()
        self._refs = [reference for reference in self._refs if reference() is not None]

    def _clear_history(self, *, eviction: bool = False) -> None:
        if eviction:
            self.report.history_evictions += len(self._history_entries) + len(self._history_pending)
            self.report.canonical_prompt_evictions += (int(self._canonical_prompt_entry is not None)
                                                      + int(self._canonical_prompt_pending is not None))
        self.discard_pending_history()
        if self._canonical_prompt_entry is not None:
            self._release_canonical_prompt_entry(self._canonical_prompt_entry)
            self._canonical_prompt_entry = None
        for entry in self._history_entries.values():
            self._release_history_output(entry.output)
        self._history_entries.clear()
        self._history_key_elements = self._history_key_bytes = 0
        self._refs = [reference for reference in self._refs if reference() is not None]

    def _key_admits(self, elements: int, key_bytes: int) -> bool:
        prompts = tuple(entry for entry in (self._canonical_prompt_entry, self._canonical_prompt_pending)
                        if entry is not None)
        return (self._entry_count + len(self._history_entries) + len(self._history_pending) + len(prompts) < MAX_ENTRIES_PER_GROUP
                and self._key_elements + self._history_key_elements + sum(entry.key_elements for entry in prompts)
                + elements <= MAX_KEY_ELEMENTS_PER_GROUP
                and self._key_bytes + self._history_key_bytes + sum(entry.key_bytes for entry in prompts)
                + key_bytes <= MAX_KEY_BYTES_PER_GROUP)

    def _release_canonical_prompt_output(self, output: packed.PackedEncoderOutput) -> None:
        size = sum(_bytes(value) for value in _packed_tensors(output))
        self.report.current_canonical_prompt_bytes -= size
        self.report.current_bytes -= size

    def _release_canonical_prompt_entry(self, entry: _CanonicalPromptEntry) -> None:
        self._release_canonical_prompt_output(entry.output)
        size = sum(_bytes(value) for value in entry.inputs)
        self.report.current_canonical_prompt_bytes -= size
        self.report.current_bytes -= size

    def _canonical_prompt_skip(self, reason: str) -> None:
        self.report.canonical_prompt_admission_skips += 1
        _bump(self.report.canonical_prompt_skip_reasons, reason)

    def _history_skip(self, reason: str) -> None:
        self.report.history_admission_skips += 1
        _bump(self.report.history_skip_reasons, reason)

    def _release_encoder_bytes(self, size: int, *, prompt: bool) -> None:
        self.report.current_bytes -= size
        if prompt:
            self.report.current_canonical_prompt_bytes -= size
        else:
            self.report.current_history_bytes -= size

    def _clone_encoder_tensors(self, values: tuple[t.Tensor, ...], *, prompt: bool) -> list[t.Tensor] | None:
        """Copy encoder state with one accounting and partial-failure cleanup path."""
        skip = self._canonical_prompt_skip if prompt else self._history_skip
        copies: list[t.Tensor] = []
        copied_bytes = 0
        retained = False
        try:
            try:
                for value in values:
                    copied = _clone_for_cache(value)
                    copies.append(copied)
                    self._refs.append(weakref.ref(copied))
                    size = _bytes(copied)
                    copied_bytes += size
                    self.report.current_bytes += size
                    self.report.peak_bytes = max(self.report.peak_bytes, self.report.current_bytes)
                    if prompt:
                        self.report.current_canonical_prompt_bytes += size
                        self.report.canonical_prompt_total_captured_bytes += size
                        self.report.peak_canonical_prompt_bytes = max(
                            self.report.peak_canonical_prompt_bytes, self.report.current_canonical_prompt_bytes
                        )
                    else:
                        self.report.current_history_bytes += size
                        self.report.history_total_captured_bytes += size
                        self.report.peak_history_bytes = max(self.report.peak_history_bytes, self.report.current_history_bytes)
                    del copied
            except t.cuda.OutOfMemoryError:
                self.report.optional_copy_ooms += 1
                if prompt:
                    self.report.canonical_prompt_clone_ooms += 1
                else:
                    self.report.history_clone_ooms += 1
                skip("optional_copy_oom")
                return None
            if any(_bytes(value) != value.numel() * value.element_size() for value in copies):
                skip("clone_layout_or_storage")
                return None
            retained = True
            return copies
        finally:
            if not retained:
                self._release_encoder_bytes(copied_bytes, prompt=prompt)
                copies.clear()
                self._refs = [reference for reference in self._refs if reference() is not None]

    def _clone_encoder_output(self, output: packed.PackedEncoderOutput, *, prompt: bool) -> packed.PackedEncoderOutput | None:
        """Own five packed tensors, including hit transients; caller releases their charge."""
        skip = self._canonical_prompt_skip if prompt else self._history_skip
        if (type(output) is not packed.PackedEncoderOutput or type(output.layout) is not packed.PackedEncoderLayout
                or output.layout.batch_size != 1
                or (prompt and (type(output.layout.n_segments) is not int or output.layout.n_segments < 1))
                or (not prompt and output.layout.n_segments != 1)):
            skip("unsupported_packed_output")
            return None
        values = _packed_tensors(output)
        if (any(type(value) is not t.Tensor or value.layout != t.strided or not value.is_cuda
                or value.device != output.values.device or not value.is_contiguous() or value.storage_offset() != 0
                or value.requires_grad or t.is_inference(value) for value in values)
                or output.values.dtype != t.bfloat16):
            skip("unsupported_packed_tensors")
            return None
        needed = sum(value.numel() * value.element_size() for value in values)
        if not self._memory_admits(needed, output.values.device):
            skip("tensor_budget_or_headroom")
            return None
        copies = self._clone_encoder_tensors(values, prompt=prompt)
        if copies is None:
            return None
        retained = False
        try:
            layout = output.layout
            cloned = packed.PackedEncoderOutput(copies[0], packed.PackedEncoderLayout(
                *copies[1:], batch_size=layout.batch_size, n_segments=layout.n_segments,
                feature_count=layout.feature_count,
            ))
            packed.mark_packed_encoder_output_dynamic(cloned)
            retained = True
            return cloned
        finally:
            if not retained:
                self._release_encoder_bytes(sum(_bytes(value) for value in copies), prompt=prompt)
            copies.clear()

    def _clone_history_output(self, output: packed.PackedEncoderOutput) -> packed.PackedEncoderOutput | None:
        return self._clone_encoder_output(output, prompt=False)

    def _clone_canonical_prompt_output(self, output: packed.PackedEncoderOutput) -> packed.PackedEncoderOutput | None:
        return self._clone_encoder_output(output, prompt=True)

    def _stage_canonical_prompt(self, inputs: tuple[t.Tensor, ...], output: packed.PackedEncoderOutput) -> None:
        """Stage one prompt per group; committed entries are never replaced/readmitted.

        A head-priority eviction permanently ends prompt reuse for this group.
        The same canonical call's postguard publishes a successful admission.
        """
        if self._canonical_prompt_committed_once or self._canonical_prompt_pending is not None:
            return
        metadata = tuple(_tensor_meta(value) for value in inputs)
        # Conservative key charges include device-resident contents, without
        # materializing those values on the host. The tensor cap is separate.
        elements = sum(value.numel() + 2 * value.ndim + 4 for value in inputs)
        input_bytes = sum(value.numel() * value.element_size() for value in inputs)
        key_bytes = input_bytes + len(repr(metadata).encode())
        if not self._key_admits(elements, key_bytes):
            self._canonical_prompt_skip("entry_or_key_budget")
            return
        needed = input_bytes + sum(value.numel() * value.element_size() for value in _packed_tensors(output))
        if not self._memory_admits(needed, inputs[0].device):
            self._canonical_prompt_skip("tensor_budget_or_headroom")
            return
        copies = self._clone_encoder_tensors(inputs, prompt=True)
        if copies is None:
            return
        cloned_output = None
        retained = False
        try:
            cloned_output = self._clone_canonical_prompt_output(output)
            if cloned_output is None:
                return
            self._canonical_prompt_pending = _CanonicalPromptEntry(
                tuple(copies), metadata, cloned_output, elements, key_bytes,
            )
            retained = True
            self.report.canonical_prompt_peak_entries = 1
            self.report.canonical_prompt_peak_key_elements = max(self.report.canonical_prompt_peak_key_elements, elements)
            self.report.canonical_prompt_peak_key_bytes = max(self.report.canonical_prompt_peak_key_bytes, key_bytes)
        finally:
            if not retained:
                if cloned_output is not None:
                    self._release_canonical_prompt_output(cloned_output)
                self._release_encoder_bytes(sum(_bytes(value) for value in copies), prompt=True)
            copies.clear()
            del cloned_output
            self._refs = [reference for reference in self._refs if reference() is not None]

    def encode_prompt_and_history(self, model: Any, inputs: tuple[t.Tensor, ...],
                                  feature_history: Sequence[ConditioningAction], *,
                                  device: t.device) -> packed.PackedEncoderOutput | None:
        """One synchronous prompt/history operation; never return an owned hit."""
        self._check_thread()
        if (not self.settings.prompt_encoder or not self.settings.history_encoder
                or model is not self._model or self._phase != "collecting" or self._guard is None
                or self._invalid_reason is not None or device != self._guard.device
                or t.is_grad_enabled() or t.is_inference_mode_enabled()
                or type(inputs) is not tuple or len(inputs) != 3
                or type(feature_history) not in (tuple, list) or len(feature_history) > MAX_ENTRIES_PER_GROUP):
            return None
        if (any(type(value) is not t.Tensor or value.dtype != dtype or value.device != device
                or value.layout != t.strided or value.ndim != 3 or value.shape[0] != 1
                or not value.is_contiguous() or value.storage_offset() != 0
                or value.requires_grad or t.is_inference(value)
                for value, dtype in zip(inputs, (t.int64, t.int64, t.bool), strict=True))
                or any(value.shape != inputs[0].shape for value in inputs[1:])):
            return None
        keys = _history_keys(feature_history)
        if keys is None:
            return None
        self.discard_pending_history()
        if feature_history:
            self.report.history_calls += 1
            self.report.history_guard_calls += 1
            valid, _ = self._check_guard()
            if not valid:
                return None
        self.report.canonical_prompt_calls += 1
        entry = self._canonical_prompt_entry
        copied = (self._clone_canonical_prompt_output(entry.output)
                  if feature_history and entry is not None and entry.matches(inputs) else None)
        try:
            if copied is None:
                self.report.canonical_prompt_misses += 1
                encoder_out = model.encoder(*inputs)
                self._stage_canonical_prompt(inputs, encoder_out)
            else:
                self.report.canonical_prompt_hits += 1
                encoder_out = copied
            if not feature_history:
                # This is always a native result, never a private defensive copy.
                return encoder_out
            return self._append_history_after_guard(model, encoder_out, feature_history, keys, device=device)
        finally:
            if copied is not None:
                self._release_canonical_prompt_output(copied)
                del copied
            self._refs = [reference for reference in self._refs if reference() is not None]

    def encode_history(self, model: Any, encoder_out: packed.PackedEncoderOutput,
                       feature_history: Sequence[ConditioningAction], *, device: t.device) -> packed.PackedEncoderOutput | None:
        """Guard and execute one synchronous collection loop, or decline before it."""
        self._check_thread()
        if (not self.settings.history_encoder or model is not self._model or self._phase != "collecting"
                or t.is_grad_enabled() or t.is_inference_mode_enabled()
                or type(feature_history) not in (tuple, list) or not feature_history
                or len(feature_history) > MAX_ENTRIES_PER_GROUP):
            return None
        keys = _history_keys(feature_history)
        if keys is None:
            return None
        self.discard_pending_history()
        self.report.history_calls += 1
        self.report.history_guard_calls += 1
        valid, _ = self._check_guard()
        if not valid or self._guard is None or device != self._guard.device:
            return None
        return self._append_history_after_guard(model, encoder_out, feature_history, keys, device=device)

    def _append_history_after_guard(self, model: Any, encoder_out: packed.PackedEncoderOutput,
                                    feature_history: Sequence[ConditioningAction],
                                    keys: list[tuple[tuple[Any, ...], int, int]], *,
                                    device: t.device) -> packed.PackedEncoderOutput:
        # Only synchronous entry points call this immediately after their guard.
        for action, parts in zip(feature_history, keys, strict=True):
            key, elements, key_bytes = parts
            inputs = self._action_tensors(action, device=device)
            self.report.history_action_calls += 1
            entry = self._history_entries.get(key)
            copied = None if entry is None else self._clone_history_output(entry.output)
            try:
                if copied is not None:
                    self.report.history_hits += 1
                    encoder_out = self._append_history(encoder_out, copied)
                    continue
                self.report.history_misses += 1
                encoded = model.encoder(*inputs)
                # Preserve native validation/append errors before optional copies.
                encoder_out = self._append_history(encoder_out, encoded)
                if key not in self._history_entries and key not in self._history_pending:
                    if not self._key_admits(elements, key_bytes):
                        self._history_skip("entry_or_key_budget")
                    else:
                        saved = self._clone_history_output(encoded)
                        if saved is not None:
                            self._history_pending[key] = _HistoryEntry(saved, elements, key_bytes)
                            self._history_key_elements += elements
                            self._history_key_bytes += key_bytes
                            self.report.history_peak_entries = max(
                                self.report.history_peak_entries, len(self._history_entries) + len(self._history_pending)
                            )
                            del saved
                del encoded
            finally:
                if copied is not None:
                    self._release_history_output(copied)
                    del copied
                self._refs = [reference for reference in self._refs if reference() is not None]
        return encoder_out

    def close(self) -> None:
        self._check_thread()
        if self._phase == "closed":
            return
        started = time.perf_counter()
        self._clear_entries()
        self._guard = None
        alive = sum(reference() is not None for reference in self._refs)
        self._refs.clear()
        self.report.owned_tensor_objects_alive_after_cleanup = alive
        self.report.max_owned_tensor_objects_alive_after_cleanup = max(
            self.report.max_owned_tensor_objects_alive_after_cleanup, alive
        )
        if not self._completed and not self._failed:
            self.report.groups_abandoned += 1
        self._phase = "closed"
        self.report.groups_closed += 1
        with _SESSION_LOCK:
            if _SESSION_OWNERS.get(self._model_identity) is not self._owner:
                raise RuntimeError("Hidden replay session ownership changed before cleanup")
            del _SESSION_OWNERS[self._model_identity]
            self._model = self._group = None
            self.report.active_groups -= 1
            self.report.cleaned_up = self.report.active_groups == 0 and self.report.current_bytes == 0
        self.report.cleanup_seconds += time.perf_counter() - started

    def _invalidate(self, reason: str) -> None:
        if self._invalid_reason is None:
            self._invalid_reason = reason
            self.report.invalidations += 1
            _bump(self.report.invalidation_reasons, reason)
            self._clear_entries()

    def _check_guard(self, *, initial: bool = False) -> tuple[bool, str]:
        if self._invalid_reason is not None:
            return False, self._invalid_reason
        self.report.guard_calls += 1
        started = time.perf_counter()
        try:
            if self._guard is None and initial:
                self._guard, reason = NativeFrozenState.capture(self._model, policy_dependencies=self._dependencies)
                valid = self._guard is not None
            elif self._guard is not None:
                valid, reason = self._guard.matches()
            else:
                valid, reason = False, "no_behavior_guard"
        finally:
            self.report.guard_seconds += time.perf_counter() - started
        if not valid:
            self._invalidate(reason)
        return valid, reason

    def _skip(self, reason: str) -> None:
        self.report.capture_skips += 1
        _bump(self.report.capture_skip_reasons, reason)

    def _miss(self, reason: str) -> None:
        self.report.replay_misses += 1
        _bump(self.report.miss_reasons, reason)

    def prepare_request(self, model: Any, prompt: Any, *, committed_prefix: Sequence[int],
                        outgoing_seg_map: Sequence[int], pool_positions: Sequence[int],
                        feature_history: Sequence[ConditioningAction],
                        train_feature_head_only: bool) -> CanonicalReplayRequest | None:
        self._check_thread()
        if model is not self._model or self._phase not in ("collecting", "replaying"):
            return None
        behavior = self._phase == "collecting"
        if behavior:
            if self._history_pending or self._canonical_prompt_pending is not None:
                self.discard_pending_history()
            self.report.behavior_calls += 1
        else:
            self.report.replay_calls += 1
        reason = "unsupported_mode"
        mode_ok = ((behavior and train_feature_head_only is False and not t.is_grad_enabled())
                   or (not behavior and train_feature_head_only is True and t.is_grad_enabled()))
        if mode_ok and type(prompt) is self._prompt_type:
            key, elements, key_bytes, reason = _key(prompt, committed_prefix, outgoing_seg_map, pool_positions, feature_history)
            if key is not None:
                values = _prompt_tensors(prompt)
                if values is None:
                    reason = "unsupported_prompt_tensors"
                else:
                    valid, reason = self._check_guard(initial=behavior)
                    if valid:
                        request = CanonicalReplayRequest(key, elements, key_bytes, values,
                            tuple(_tensor_meta(value) for value in values), len(committed_prefix), self._owner)
                        if not behavior:
                            return request
                        if not self._key_admits(elements, key_bytes):
                            self._clear_history(eviction=True)
                        if not self._key_admits(elements, key_bytes):
                            reason = "entry_or_key_budget"
                        elif any(entry.matches(request) for entry in self._entries.get(key, ())):
                            reason = "duplicate_behavior_state"
                        else:
                            return request
        if behavior:
            self._skip(reason)
        else:
            self._miss(reason)
        return None

    def lookup(self, request: CanonicalReplayRequest) -> CachedHeadInput | None:
        self._check_thread()
        if self._phase != "replaying" or request.owner is not self._owner:
            raise RuntimeError("Hidden replay lookup requires its replay request")
        for entry in self._entries.get(request.key, ()):
            if entry.matches(request):
                self.report.replay_hits += 1
                return CachedHeadInput(entry.hidden, entry.encoder_slots)
        self._miss("missing_state_or_prompt_content")
        return None

    def _memory_admits(self, needed: int, device: t.device) -> bool:
        if self.report.current_bytes + needed > self.settings.max_cache_bytes:
            return False
        free, _ = t.cuda.mem_get_info(device)
        reusable = max(0, t.cuda.memory_reserved(device) - t.cuda.memory_allocated(device))
        self.report.admission_free_bytes = free
        self.report.admission_reusable_reserved_bytes = reusable
        self.report.admission_estimated_available_bytes = free + reusable
        return free + reusable >= needed + self.settings.min_free_bytes

    def capture(self, request: CanonicalReplayRequest, *, hidden: t.Tensor, encoder_slots: int) -> None:
        self._check_thread()
        if self._phase != "collecting" or request.owner is not self._owner:
            raise RuntimeError("Hidden replay capture requires its collection request")
        self.report.capture_attempts += 1
        valid, reason = self._check_guard()
        if not valid:
            self._skip(reason)
            return
        # The same canonical call has now completed native compute against the
        # original guard baseline. Pending history is independently valid even
        # if the optional head-input copy below is declined.
        self.report.history_admitted_entries += len(self._history_pending)
        self._history_entries.update(self._history_pending)
        self._history_pending.clear()
        if self._canonical_prompt_pending is not None:
            assert self._canonical_prompt_entry is None
            self._canonical_prompt_entry = self._canonical_prompt_pending
            self._canonical_prompt_pending = None
            self._canonical_prompt_committed_once = True
            self.report.canonical_prompt_admitted_entries += 1
        if (type(hidden) is not t.Tensor or hidden.ndim != 3 or hidden.shape[:2] != (1, request.prefix_length)
                or not hidden.is_cuda or hidden.dtype != t.bfloat16 or not hidden.is_contiguous()
                or hidden.storage_offset() != 0 or hidden.requires_grad or t.is_inference(hidden)
                or t.is_grad_enabled() or type(encoder_slots) is not int):
            self._skip("unsupported_hidden")
            return
        values = request.prompt_tensors
        if (any(value.is_cuda and value.device != hidden.device for value in values)
                or tuple(_tensor_meta(value) for value in values) != request.prompt_metadata):
            self._skip("prompt_metadata_changed_during_capture")
            return
        needed = hidden.numel() * hidden.element_size() + sum(value.numel() * value.element_size() for value in values)
        # History admissions happened after prepare_request. Recheck all caps,
        # giving existing hidden replay priority over this optional memo.
        key_admitted = self._key_admits(request.key_elements, request.key_bytes)
        memory_admitted = self._memory_admits(needed, hidden.device)
        if (not key_admitted or not memory_admitted) and (
                self._history_entries or self._history_pending or self._canonical_prompt_entry is not None
                or self._canonical_prompt_pending is not None):
            self._clear_history(eviction=True)
            key_admitted = self._key_admits(request.key_elements, request.key_bytes)
            memory_admitted = self._memory_admits(needed, hidden.device)
        if not key_admitted:
            self._skip("entry_or_key_budget")
            return
        if not memory_admitted:
            self._skip("tensor_budget_or_headroom")
            return
        started = time.perf_counter()
        copies: list[t.Tensor] = []
        retained = False
        hidden_bytes = prompt_bytes = 0
        try:
            try:
                for index, value in enumerate((hidden, *values)):
                    copied = _clone_for_cache(value)
                    copies.append(copied)
                    self._refs.append(weakref.ref(copied))
                    size = _bytes(copied)
                    if index == 0:
                        hidden_bytes += size
                        self.report.current_hidden_bytes += size
                    else:
                        prompt_bytes += size
                        self.report.current_prompt_bytes += size
                    self.report.current_bytes += size
                    self.report.total_captured_bytes += size
                    self.report.peak_bytes = max(self.report.peak_bytes, self.report.current_bytes)
                    self.report.peak_hidden_bytes = max(self.report.peak_hidden_bytes, self.report.current_hidden_bytes)
                    del copied
            except t.cuda.OutOfMemoryError:
                self.report.optional_copy_ooms += 1
                self._skip("optional_copy_oom")
                return
            if (not copies[0].is_contiguous() or tuple(copies[0].stride()) != tuple(hidden.stride())
                    or any(_bytes(value) != value.numel() * value.element_size() for value in copies)):
                self._skip("clone_layout_or_storage")
                return
            entry = _Entry(copies[0], tuple(copies[1:]), request.prompt_metadata, encoder_slots)
            self._entries.setdefault(request.key, []).append(entry)
            retained = True
            self._entry_count += 1
            self._key_elements += request.key_elements
            self._key_bytes += request.key_bytes
            self.report.admitted_entries += 1
            self.report.peak_entries = max(self.report.peak_entries, self._entry_count)
            self.report.peak_key_elements = max(self.report.peak_key_elements, self._key_elements)
            self.report.peak_key_bytes = max(self.report.peak_key_bytes, self._key_bytes)
            _bump(self.report.prefix_length_counts, str(request.prefix_length), limit=MAX_PREFIX_KEYS)
            if not self._admitted:
                self._admitted = True
                self.report.admitted_groups += 1
                self.report.applied = True
        finally:
            if not retained:
                self.report.current_hidden_bytes -= hidden_bytes
                self.report.current_prompt_bytes -= prompt_bytes
                self.report.current_bytes -= hidden_bytes + prompt_bytes
            copies.clear()
            self._refs = [reference for reference in self._refs if reference() is not None]
            self.report.capture_seconds += time.perf_counter() - started
