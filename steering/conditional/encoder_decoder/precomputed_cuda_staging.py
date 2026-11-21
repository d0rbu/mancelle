"""Bounded CPU lookahead and stream-safe CUDA staging for packed batches.

This deliberately stages only the fixed 10-tuple from
``precomputed_data_loader(device="cpu")``. It is opt-in for verified remote
packed training; local, live, validation, and legacy prefetch paths do not use
it.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch as t
from beartype import beartype
from loguru import logger

from steering.refined_types import PositiveIntLike


class PrecomputedCudaStagingError(RuntimeError):
    """Invalid input, transfer failure, or worker lifecycle failure."""


type PrecomputedBatch = tuple[
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor,
    t.Tensor | None,
    dict[str, Any],
]


@dataclass(frozen=True)
class CudaStagingStats:
    produced_batches: int
    staged_batches: int
    consumed_batches: int
    host_queue_bytes: int
    host_queue_peak_bytes: int
    producer_blocked_seconds: float
    h2d_seconds: float
    consumer_wait_seconds: float


@dataclass(frozen=True)
class _TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: t.dtype


@dataclass(frozen=True)
class _BatchSchema:
    tensors: tuple[_TensorSpec | None, ...]
    state_keys: frozenset[str]


@dataclass(frozen=True)
class _Data:
    batch: object
    byte_size: int = 0
    event: t.cuda.Event | None = None
    slot_index: int = -1


@dataclass(frozen=True)
class _End:
    pass


@dataclass(frozen=True)
class _Failure:
    error: BaseException


type _Message = _Data | _End | _Failure
type _PinnedSlot = tuple[t.Tensor | None, ...]


class _StagingStopped(Exception):
    pass


class _ByteBoundedQueue:
    """FIFO whose data capacity is measured in retained tensor-storage bytes."""

    def __init__(self, capacity_bytes: int, stop_event: threading.Event) -> None:
        self.capacity_bytes = capacity_bytes
        self._stop_event = stop_event
        self._condition = threading.Condition()
        self._items: deque[_Message] = deque()
        self._bytes = 0
        self._peak_bytes = 0

    @property
    def current_bytes(self) -> int:
        with self._condition:
            return self._bytes

    @property
    def peak_bytes(self) -> int:
        with self._condition:
            return self._peak_bytes

    def put_data(self, batch: object, byte_size: int) -> float:
        if byte_size > self.capacity_bytes:
            raise PrecomputedCudaStagingError(
                "one precomputed CPU batch exceeds remote CUDA staging "
                f"host_queue_bytes: batch={byte_size}, cap={self.capacity_bytes}"
            )
        started = time.perf_counter()
        with self._condition:
            while self._bytes + byte_size > self.capacity_bytes:
                if self._stop_event.is_set():
                    raise _StagingStopped
                self._condition.wait(timeout=0.1)
            if self._stop_event.is_set():
                raise _StagingStopped
            self._items.append(_Data(batch, byte_size=byte_size))
            self._bytes += byte_size
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            self._condition.notify_all()
        return time.perf_counter() - started

    def put_control(self, message: _End | _Failure) -> None:
        with self._condition:
            if not self._stop_event.is_set():
                self._items.append(message)
                self._condition.notify_all()

    def get(self) -> _Message:
        with self._condition:
            while not self._items:
                if self._stop_event.is_set():
                    raise _StagingStopped
                self._condition.wait(timeout=0.1)
            message = self._items.popleft()
            if isinstance(message, _Data):
                self._bytes -= message.byte_size
            self._condition.notify_all()
            return message

    def stop(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._items.clear()
            self._bytes = 0
            self._condition.notify_all()


@dataclass
class _MutableStats:
    produced_batches: int = 0
    staged_batches: int = 0
    consumed_batches: int = 0
    producer_blocked_seconds: float = 0.0
    h2d_seconds: float = 0.0
    consumer_wait_seconds: float = 0.0


@dataclass
class _Runtime:
    inner: Iterator[object]
    device: t.device
    expected_batch_size: int
    expected_context_len: int
    expected_pad_f: int
    expected_n_seg: int | None
    host_queue: _ByteBoundedQueue
    ready_queue: queue.Queue[_Message]
    free_slots: queue.Queue[int]
    slots: list[_PinnedSlot | None]
    stop_event: threading.Event
    stats: _MutableStats = field(default_factory=_MutableStats)
    stats_lock: threading.Lock = field(default_factory=threading.Lock)
    close_lock: threading.Lock = field(default_factory=threading.Lock)
    schema: _BatchSchema | None = None
    threads: list[threading.Thread] = field(default_factory=list)
    closed: bool = False


# Warm issue-#842 traces reached roughly 90-second shard tails. A worker gets
# more than three times that observed tail to unwind before shutdown fails.
_JOIN_TIMEOUT_SECONDS = 300.0


def validate_precomputed_cuda_staging_eligibility(
    *,
    device: str | t.device,
    label_dropout: object,
    selection_dir: object,
    remote_materializer: object,
    packed_materializer: object,
    resume_state_dict: Mapping[str, Any] | None,
    expected_batch_size: object,
    expected_context_len: object,
    expected_pad_f: object,
    expected_n_seg: object,
) -> t.device:
    """Reject every route except static verified remote packed CUDA training."""
    resolved = t.device(device)
    if resolved.type != "cuda" or resolved.index is None:
        raise PrecomputedCudaStagingError(
            f"remote CUDA staging requires an explicitly indexed CUDA device, got {resolved}"
        )
    if label_dropout != 0.0:
        raise PrecomputedCudaStagingError(
            "remote CUDA staging requires label_dropout=0.0 because host "
            "lookahead must not advance the process-global dropout RNG"
        )
    if selection_dir is None or remote_materializer is None or packed_materializer is None:
        raise PrecomputedCudaStagingError(
            "remote CUDA staging requires verified remote packed and selection "
            "materializers plus an explicit selection_dir; fused/local paths are forbidden"
        )
    for name, value in (
        ("device_batch_size", expected_batch_size),
        ("max_seq_len", expected_context_len),
        ("pad_F", expected_pad_f),
        ("max_n_seg", expected_n_seg),
    ):
        if type(value) is not int or value <= 0:
            raise PrecomputedCudaStagingError(
                f"remote CUDA staging requires a positive static {name}, got {value!r}"
            )
    if resume_state_dict is not None and "label_dropout_rng_state" in resume_state_dict:
        raise PrecomputedCudaStagingError(
            "remote CUDA staging refuses resume state containing "
            "label_dropout_rng_state; use a checkpoint produced with label_dropout=0.0"
        )
    return resolved


def _walk_tensors(value: object) -> Iterator[t.Tensor]:
    if isinstance(value, t.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_tensors(key)
            yield from _walk_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)


def retained_tensor_bytes(value: object) -> int:
    """Count unique backing-storage bytes retained by a nested batch."""
    seen: set[tuple[str, int | None, int, int]] = set()
    total = 0
    for tensor in _walk_tensors(value):
        storage = tensor.untyped_storage()
        key = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _validate_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: t.dtype,
) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise PrecomputedCudaStagingError(f"batch field {name} must be a tensor")
    if value.device.type != "cpu":
        raise PrecomputedCudaStagingError(f"batch field {name} must be on CPU, got {value.device}")
    if (
        value.layout != t.strided
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.requires_grad
    ):
        raise PrecomputedCudaStagingError(
            f"batch field {name} must be contiguous, strided, gradient-free, "
            "and have storage_offset=0"
        )
    if tuple(value.shape) != shape:
        raise PrecomputedCudaStagingError(
            f"batch field {name} has shape {tuple(value.shape)}, expected {shape}"
        )
    if value.dtype != dtype:
        raise PrecomputedCudaStagingError(
            f"batch field {name} has dtype {value.dtype}, expected {dtype}"
        )
    return value


def validate_cpu_precomputed_batch(
    batch: object,
    *,
    expected_batch_size: int,
    expected_context_len: int,
    expected_pad_f: int,
    expected_n_seg: int | None = None,
) -> tuple[PrecomputedBatch, _BatchSchema]:
    """Validate and describe the packed loader's fixed CPU batch contract."""
    if not isinstance(batch, tuple) or len(batch) != 10:
        raise PrecomputedCudaStagingError(
            f"staging requires the packed loader's exact 10-tuple, got {type(batch).__name__}"
        )
    raw_state = batch[9]
    if not isinstance(raw_state, dict) or not all(isinstance(key, str) for key in raw_state):
        raise PrecomputedCudaStagingError("batch state must be a string-keyed dict")
    state = cast(dict[str, Any], raw_state)
    if "label_dropout_rng_state" in state:
        raise PrecomputedCudaStagingError(
            "staged batch unexpectedly contains label_dropout_rng_state"
        )
    if any(tensor.device.type != "cpu" or tensor.requires_grad for tensor in _walk_tensors(state)):
        raise PrecomputedCudaStagingError("batch state tensors must be on CPU and gradient-free")
    if not isinstance(batch[2], t.Tensor) or batch[2].ndim != 3:
        raise PrecomputedCudaStagingError("enc_fids must have shape [B, n_seg, F]")

    B, T = expected_batch_size, expected_context_len
    n_seg = int(batch[2].shape[1])
    if n_seg <= 0:
        raise PrecomputedCudaStagingError("staging requires a positive static n_seg")
    if expected_n_seg is not None and n_seg != expected_n_seg:
        raise PrecomputedCudaStagingError(
            f"batch n_seg={n_seg}, expected manifest max_n_seg={expected_n_seg}"
        )
    feature_shape = (B, n_seg, expected_pad_f)
    fields = (
        ("inputs", batch[0], (B, T), t.int64),
        ("targets", batch[1], (B, T), t.int64),
        ("enc_fids", batch[2], feature_shape, t.int64),
        ("enc_mags", batch[3], feature_shape, t.int64),
        ("enc_mask", batch[4], feature_shape, t.bool),
        ("enc_pcts", batch[5], feature_shape, t.float32),
        ("seg_map", batch[6], (B, T), t.int64),
        ("seg_doc_id", batch[7], (B, n_seg), t.int64),
    )
    tensors = tuple(
        _validate_tensor(value, name=name, shape=shape, dtype=dtype)
        for name, value, shape, dtype in fields
    )
    fp_targets: t.Tensor | None = None
    if batch[8] is not None:
        raw_fp = batch[8]
        if not isinstance(raw_fp, t.Tensor) or raw_fp.ndim != 3:
            raise PrecomputedCudaStagingError("fp_targets must be None or have shape [B, n_seg, V]")
        fp_width = int(raw_fp.shape[2])
        if fp_width <= 0:
            raise PrecomputedCudaStagingError("staging requires a positive FP-target width")
        fp_targets = _validate_tensor(
            raw_fp,
            name="fp_targets",
            shape=(B, n_seg, fp_width),
            dtype=t.float32,
        )
    typed = cast(PrecomputedBatch, (*tensors, fp_targets, state))
    schema = _BatchSchema(
        tensors=tuple(
            (
                None
                if tensor is None
                else _TensorSpec(
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                )
            )
            for tensor in typed[:9]
        ),
        state_keys=frozenset(state),
    )
    return typed, schema


def _assert_same_schema(expected: _BatchSchema, actual: _BatchSchema) -> None:
    if actual != expected:
        raise PrecomputedCudaStagingError(
            f"staging requires one static CPU batch schema; expected={expected!r}, got={actual!r}"
        )


def _allocate_slot(schema: _BatchSchema) -> _PinnedSlot:
    return tuple(
        (
            None
            if spec is None
            else t.empty_strided(
                spec.shape,
                spec.stride,
                dtype=spec.dtype,
                device="cpu",
                pin_memory=True,
            )
        )
        for spec in schema.tensors
    )


def _stage_batch(
    slot: _PinnedSlot,
    batch: PrecomputedBatch,
    *,
    device: t.device,
) -> PrecomputedBatch:
    staged: list[t.Tensor | None] = []
    for pinned, source in zip(slot, batch[:9], strict=True):
        if pinned is None or source is None:
            if pinned is not source:
                raise PrecomputedCudaStagingError("FP-target presence changed")
            staged.append(None)
            continue
        pinned.copy_(source, non_blocking=False)
        destination = t.empty_strided(
            pinned.shape,
            pinned.stride(),
            dtype=pinned.dtype,
            device=device,
        )
        destination.copy_(pinned, non_blocking=True)
        staged.append(destination)
    return cast(PrecomputedBatch, (*staged, batch[9]))


def _put_ready(runtime: _Runtime, message: _Message) -> None:
    while not runtime.stop_event.is_set():
        try:
            runtime.ready_queue.put(message, timeout=0.1)
            return
        except queue.Full:
            pass
    raise _StagingStopped


def _get_free_slot(runtime: _Runtime) -> int:
    while not runtime.stop_event.is_set():
        try:
            return runtime.free_slots.get(timeout=0.1)
        except queue.Empty:
            pass
    raise _StagingStopped


def _producer_main(runtime: _Runtime) -> None:
    try:
        while not runtime.stop_event.is_set():
            try:
                batch = next(runtime.inner)
            except StopIteration:
                runtime.host_queue.put_control(_End())
                return
            blocked = runtime.host_queue.put_data(
                batch,
                retained_tensor_bytes(batch),
            )
            with runtime.stats_lock:
                runtime.stats.produced_batches += 1
                runtime.stats.producer_blocked_seconds += blocked
    except _StagingStopped:
        pass
    except BaseException as error:
        runtime.host_queue.put_control(_Failure(error))


def _h2d_main(runtime: _Runtime) -> None:
    slot_index: int | None = None
    stream: Any = None
    try:
        with t.cuda.device(runtime.device):
            t.cuda.set_device(runtime.device)
            stream = t.cuda.Stream(device=runtime.device)
            while not runtime.stop_event.is_set():
                message = runtime.host_queue.get()
                if isinstance(message, _End):
                    _put_ready(runtime, message)
                    return
                if isinstance(message, _Failure):
                    _put_ready(runtime, message)
                    return
                batch, schema = validate_cpu_precomputed_batch(
                    message.batch,
                    expected_batch_size=runtime.expected_batch_size,
                    expected_context_len=runtime.expected_context_len,
                    expected_pad_f=runtime.expected_pad_f,
                    expected_n_seg=runtime.expected_n_seg,
                )
                if runtime.schema is None:
                    runtime.schema = schema
                else:
                    _assert_same_schema(runtime.schema, schema)
                slot_index = _get_free_slot(runtime)
                slot = runtime.slots[slot_index]
                if slot is None:
                    slot = _allocate_slot(schema)
                    runtime.slots[slot_index] = slot
                started = time.perf_counter()
                with t.cuda.stream(stream):
                    staged = _stage_batch(slot, batch, device=runtime.device)
                    event = t.cuda.Event(blocking=False)
                    event.record(stream)
                # This is a per-event CPU wait, never a process-wide CUDA sync.
                # It proves the reusable pinned source is no longer in flight
                # and makes queued-batch destruction safe during early close.
                event.synchronize()
                with runtime.stats_lock:
                    runtime.stats.staged_batches += 1
                    runtime.stats.h2d_seconds += time.perf_counter() - started
                _put_ready(runtime, _Data(staged, event=event, slot_index=slot_index))
                slot_index = None
    except _StagingStopped:
        pass
    except BaseException as error:
        if slot_index is not None and stream is not None:
            try:
                # A mid-batch allocation/copy failure may follow earlier async
                # copies. Order their reads before the pinned lease is released.
                stream.synchronize()
            except BaseException as synchronize_error:
                synchronize_error.add_note(f"preceded by CUDA staging failure: {error!r}")
                error = synchronize_error
        try:
            _put_ready(runtime, _Failure(error))
        except _StagingStopped:
            pass
    finally:
        if slot_index is not None:
            try:
                runtime.free_slots.put_nowait(slot_index)
            except queue.Full:
                pass


def _stop_runtime(runtime: _Runtime, *, raise_on_timeout: bool) -> None:
    with runtime.close_lock:
        if runtime.closed:
            return
        runtime.stop_event.set()
        runtime.host_queue.stop()
        deadline = time.monotonic() + _JOIN_TIMEOUT_SECONDS
        current = threading.current_thread()
        for thread in runtime.threads:
            if thread is not current and thread.ident is not None:
                thread.join(timeout=max(deadline - time.monotonic(), 0.0))
        alive = [thread.name for thread in runtime.threads if thread.is_alive()]
        if alive:
            message = f"staging workers did not stop within {_JOIN_TIMEOUT_SECONDS:.0f}s: {alive}"
            if raise_on_timeout:
                raise PrecomputedCudaStagingError(message)
            logger.error(message)
            return

        close_error: BaseException | None = None
        close = getattr(runtime.inner, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                close_error = error
        runtime.host_queue.clear()
        while True:
            try:
                runtime.ready_queue.get_nowait()
            except queue.Empty:
                break
        runtime.slots.clear()
        runtime.closed = True
        if close_error is not None:
            raise close_error


def _finalize_runtime(runtime: _Runtime) -> None:
    try:
        _stop_runtime(runtime, raise_on_timeout=False)
    except BaseException:
        logger.exception("precomputed CUDA staging finalizer failed")


class PrecomputedCudaStagingLoader(Iterator[PrecomputedBatch]):
    """Stage CPU packed batches through bounded lookahead and pinned leases."""

    @beartype
    def __init__(
        self,
        inner_loader: Iterator[object],
        *,
        device: str | t.device,
        host_queue_bytes: PositiveIntLike,
        pinned_slots: PositiveIntLike,
        expected_batch_size: PositiveIntLike,
        expected_context_len: PositiveIntLike,
        expected_pad_f: PositiveIntLike,
        expected_n_seg: PositiveIntLike | None = None,
        remote_read_cancel_event: threading.Event | None = None,
    ) -> None:
        resolved = t.device(device)
        if resolved.type != "cuda" or resolved.index is None:
            raise PrecomputedCudaStagingError(
                f"staging requires an explicitly indexed CUDA device, got {resolved}"
            )
        stop_event = (
            remote_read_cancel_event if remote_read_cancel_event is not None else threading.Event()
        )
        free_slots: queue.Queue[int] = queue.Queue(maxsize=pinned_slots)
        for slot_index in range(pinned_slots):
            free_slots.put_nowait(slot_index)
        runtime = _Runtime(
            inner=iter(inner_loader),
            device=resolved,
            expected_batch_size=expected_batch_size,
            expected_context_len=expected_context_len,
            expected_pad_f=expected_pad_f,
            expected_n_seg=expected_n_seg,
            host_queue=_ByteBoundedQueue(host_queue_bytes, stop_event),
            ready_queue=queue.Queue(maxsize=pinned_slots),
            free_slots=free_slots,
            slots=[None] * pinned_slots,
            stop_event=stop_event,
        )
        self._runtime = runtime
        self._finalizer = weakref.finalize(self, _finalize_runtime, runtime)
        runtime.threads = [
            threading.Thread(
                target=_h2d_main,
                args=(runtime,),
                name=f"packed-h2d-{resolved.index}",
                daemon=True,
            ),
            threading.Thread(
                target=_producer_main,
                args=(runtime,),
                name=f"packed-host-{resolved.index}",
                daemon=True,
            ),
        ]
        try:
            for thread in runtime.threads:
                thread.start()
        except BaseException:
            self.close()
            raise
        logger.info(
            "Remote packed CUDA staging enabled: "
            f"device={resolved}, host_queue_bytes={host_queue_bytes}, "
            f"pinned_slots={pinned_slots}"
        )

    def __iter__(self) -> PrecomputedCudaStagingLoader:
        return self

    def __next__(self) -> PrecomputedBatch:
        runtime = self._runtime
        if runtime.closed:
            raise StopIteration
        started = time.perf_counter()
        while True:
            try:
                message = runtime.ready_queue.get(timeout=0.1)
                break
            except queue.Empty:
                if runtime.stop_event.is_set() and not any(
                    thread.is_alive() for thread in runtime.threads
                ):
                    self.close()
                    raise StopIteration from None
        with runtime.stats_lock:
            runtime.stats.consumer_wait_seconds += time.perf_counter() - started
        if isinstance(message, _End):
            self.close()
            raise StopIteration
        if isinstance(message, _Failure):
            try:
                self.close()
            except BaseException as close_error:
                raise close_error from message.error
            raise message.error.with_traceback(message.error.__traceback__)

        event = message.event
        if event is None or message.slot_index < 0:
            self.close()
            raise PrecomputedCudaStagingError("malformed staged-data message")
        batch = cast(PrecomputedBatch, message.batch)
        try:
            try:
                consumer_stream = t.cuda.current_stream(runtime.device)
                consumer_stream.wait_event(event)
                for tensor in _walk_tensors(batch):
                    if tensor.device.type == "cuda":
                        tensor.record_stream(consumer_stream)
            finally:
                runtime.free_slots.put_nowait(message.slot_index)
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                raise close_error from error
            raise
        with runtime.stats_lock:
            runtime.stats.consumed_batches += 1
        return batch

    def stats(self) -> CudaStagingStats:
        runtime = self._runtime
        with runtime.stats_lock:
            return CudaStagingStats(
                produced_batches=runtime.stats.produced_batches,
                staged_batches=runtime.stats.staged_batches,
                consumed_batches=runtime.stats.consumed_batches,
                host_queue_bytes=runtime.host_queue.current_bytes,
                host_queue_peak_bytes=runtime.host_queue.peak_bytes,
                producer_blocked_seconds=runtime.stats.producer_blocked_seconds,
                h2d_seconds=runtime.stats.h2d_seconds,
                consumer_wait_seconds=runtime.stats.consumer_wait_seconds,
            )

    def close(self) -> None:
        if self._runtime.closed:
            return
        _stop_runtime(self._runtime, raise_on_timeout=True)
        self._finalizer.detach()
        stats = self.stats()
        logger.info(
            "Remote packed CUDA staging closed: "
            f"produced={stats.produced_batches}, staged={stats.staged_batches}, "
            f"consumed={stats.consumed_batches}, "
            f"host_peak_bytes={stats.host_queue_peak_bytes}, "
            f"producer_blocked_seconds={stats.producer_blocked_seconds:.3f}, "
            f"h2d_seconds={stats.h2d_seconds:.3f}, "
            f"consumer_wait_seconds={stats.consumer_wait_seconds:.3f}"
        )

    def is_terminal(self) -> bool:
        """Return whether both owned staging threads have definitely exited."""
        return not any(thread.is_alive() for thread in self._runtime.threads)
