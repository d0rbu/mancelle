"""SFT-specific contracts layered on the shared closable batch prefetcher."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest
import torch as t

from scripts.train import conditional_encdec_sft as sft
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _resolve_sft_prefetch_batches,
    _sft_step_batch_iterator,
)
from steering.profiling.markers import TimingRecorder, timed_region


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (1, 1), (8, 8)])
def test_prefetch_depth_parser_accepts_strict_non_negative_ints(
    value: int,
    expected: int,
) -> None:
    assert _resolve_sft_prefetch_batches(value) == expected


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "1", None])
def test_prefetch_depth_parser_rejects_coercible_or_negative_values(value: object) -> None:
    with pytest.raises(ValueError, match="strict non-negative int"):
        _resolve_sft_prefetch_batches(value)


def test_buffered_batches_do_not_publish_checkpoint_visible_state() -> None:
    producer = SFTState()
    consumer = SFTState()
    third_built = threading.Event()

    def batches():
        for value in range(1, 5):
            producer.consumed = value
            producer.dataloader_state = {
                "consumed": value,
                "buffered_cursor_positions": [value + 1],
            }
            producer.dataset_size = 10
            producer.artifact_coverage = {"nested": {"value": value}}
            if value == 3:
                third_built.set()
            yield (t.tensor([value]),)

    with _sft_step_batch_iterator(
        batches(),
        producer_state=producer,
        consumer_state=consumer,
        batch_count=4,
        prefetch_batches=2,
        cuda_device=None,
    ) as prefetched:
        assert third_built.wait(timeout=2.0)
        assert consumer == SFTState()

        first = next(prefetched)
        assert t.equal(first[0], t.tensor([1]))
        assert consumer.consumed == 1
        assert consumer.dataloader_state == {
            "consumed": 1,
            "buffered_cursor_positions": [2],
        }
        assert consumer.dataset_size == 10
        assert consumer.artifact_coverage == {"nested": {"value": 1}}


def test_state_snapshot_is_deeply_isolated_from_producer_mutation() -> None:
    producer = SFTState()
    consumer = SFTState()
    mutation_complete = threading.Event()

    def batches():
        producer.consumed = 1
        producer.dataloader_state = {
            "consumed": 1,
            "buffered_cursor_positions": [2],
        }
        producer.artifact_coverage = {"nested": {"value": 1}}
        yield ("first",)
        assert producer.dataloader_state is not None
        producer.dataloader_state["buffered_cursor_positions"].append(999)
        assert producer.artifact_coverage is not None
        producer.artifact_coverage["nested"]["value"] = 999
        mutation_complete.set()
        yield ("second",)

    with _sft_step_batch_iterator(
        batches(),
        producer_state=producer,
        consumer_state=consumer,
        batch_count=2,
        prefetch_batches=2,
        cuda_device=None,
    ) as prefetched:
        assert mutation_complete.wait(timeout=2.0)
        assert next(prefetched) == ("first",)
        assert consumer.dataloader_state == {
            "consumed": 1,
            "buffered_cursor_positions": [2],
        }
        assert consumer.artifact_coverage == {"nested": {"value": 1}}


def test_cuda_stream_context_failure_reaches_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def broken_stream(_stream: object):
        raise RuntimeError("stream context failed")
        yield  # pragma: no cover - makes this a contextmanager generator

    monkeypatch.setattr(sft.t.cuda, "Stream", lambda *, device: object())
    monkeypatch.setattr(sft.t.cuda, "stream", broken_stream)
    with pytest.raises(RuntimeError, match="stream context failed"):
        with _sft_step_batch_iterator(
            iter((("unreachable",),)),
            producer_state=SFTState(),
            consumer_state=SFTState(),
            batch_count=1,
            prefetch_batches=1,
            cuda_device="cuda:0",
        ) as prefetched:
            next(prefetched)


def test_step_iterator_never_produces_across_optimizer_step_boundary() -> None:
    producer = SFTState()
    consumer = SFTState()
    produced: list[int] = []

    def batches():
        for value in range(1, 6):
            produced.append(value)
            producer.consumed = value
            yield (value,)

    inner = batches()
    with _sft_step_batch_iterator(
        inner,
        producer_state=producer,
        consumer_state=consumer,
        batch_count=3,
        prefetch_batches=2,
        cuda_device=None,
    ) as step_batches:
        assert list(step_batches) == [(1,), (2,), (3,)]

    assert produced == [1, 2, 3]
    assert producer.consumed == consumer.consumed == 3
    assert next(inner) == (4,)


def test_step_iterator_rejects_negative_batch_count() -> None:
    with pytest.raises(ValueError, match="batch_count must be non-negative"):
        with _sft_step_batch_iterator(
            iter(()),
            producer_state=SFTState(),
            consumer_state=SFTState(),
            batch_count=-1,
            prefetch_batches=1,
            cuda_device=None,
        ):
            pass


def test_worker_timing_rows_publish_with_their_delivered_batch() -> None:
    producer = SFTState()
    consumer = SFTState()
    recorder = TimingRecorder()
    third_built = threading.Event()

    def batches():
        for value in (1, 2, 3):
            with timed_region("sft/worker_phase", sync_cuda=False):
                producer.consumed = value
                if value == 3:
                    third_built.set()
            yield (value,)

    with _sft_step_batch_iterator(
        batches(),
        producer_state=producer,
        consumer_state=consumer,
        batch_count=3,
        prefetch_batches=2,
        cuda_device=None,
        timing_recorder=recorder,
        timing_rank=3,
        timing_step=7,
        first_micro_step=1,
    ) as prefetched:
        assert third_built.wait(timeout=2.0)
        assert recorder.rows() == []

        assert next(prefetched) == (1,)
        first_rows = recorder.rows()
        assert len(first_rows) == 1
        assert first_rows[0]["name"] == "sft/worker_phase"
        assert first_rows[0]["rank"] == 3
        assert first_rows[0]["step"] == 7
        assert first_rows[0]["micro_step"] == 1
        assert first_rows[0]["elapsed_ns"] > 0
        assert consumer.consumed == 1

        assert next(prefetched) == (2,)
        assert [row["micro_step"] for row in recorder.rows()] == [1, 2]
        assert consumer.consumed == 2

    assert [row["micro_step"] for row in recorder.rows()] == [1, 2]
    assert consumer.consumed == 2


def test_timing_disabled_prefetch_does_not_construct_a_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedTimingRecorder:
        def __init__(self) -> None:
            raise AssertionError("disabled prefetch constructed a TimingRecorder")

    monkeypatch.setattr(sft, "TimingRecorder", UnexpectedTimingRecorder)
    producer = SFTState()
    consumer = SFTState()

    def batches():
        producer.consumed = 1
        yield ("unchanged",)

    with _sft_step_batch_iterator(
        batches(),
        producer_state=producer,
        consumer_state=consumer,
        batch_count=1,
        prefetch_batches=1,
        cuda_device=None,
    ) as prefetched:
        assert next(prefetched) == ("unchanged",)

    assert consumer.consumed == 1


def test_failed_worker_batch_does_not_publish_partial_timing_rows() -> None:
    recorder = TimingRecorder()

    def broken_batches():
        with timed_region("sft/failed_worker_phase", sync_cuda=False):
            raise RuntimeError("batch failed")
        yield ("unreachable",)  # pragma: no cover - makes this a generator

    with pytest.raises(RuntimeError, match="batch failed"):
        with _sft_step_batch_iterator(
            broken_batches(),
            producer_state=SFTState(),
            consumer_state=SFTState(),
            batch_count=1,
            prefetch_batches=1,
            cuda_device=None,
            timing_recorder=recorder,
            timing_rank=0,
            timing_step=4,
        ) as prefetched:
            next(prefetched)

    assert recorder.rows() == []


def test_cuda_event_is_recorded_by_producer_and_waited_by_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FakeEvent:
        def __init__(self, *, blocking: bool) -> None:
            calls.append(("create", blocking))

        def record(self, stream: object) -> None:
            calls.append(("record", stream))

        def wait(self, stream: object) -> None:
            calls.append(("wait", stream))

    class FakeCudaTensor:
        is_cuda = True

        def record_stream(self, stream: object) -> None:
            calls.append(("record_stream", stream))

    producer_stream = object()
    consumer_stream = object()

    def make_stream(*, device: object) -> object:
        calls.append(("create_stream", device))
        return producer_stream

    @contextmanager
    def use_stream(stream: object):
        calls.append(("stream_enter", stream))
        try:
            yield
        finally:
            calls.append(("stream_exit", stream))

    monkeypatch.setattr(sft.t.cuda, "Stream", make_stream)
    monkeypatch.setattr(sft.t.cuda, "stream", use_stream)
    monkeypatch.setattr(sft.t.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        sft.t.cuda,
        "current_stream",
        lambda device=None: (consumer_stream, device),
    )
    producer = SFTState()
    consumer = SFTState()
    tensor = FakeCudaTensor()

    def batches():
        producer.consumed = 1
        yield (tensor,)

    with _sft_step_batch_iterator(
        batches(),
        producer_state=producer,
        consumer_state=consumer,
        batch_count=1,
        prefetch_batches=1,
        cuda_device="cuda:1",
    ) as prefetched:
        assert next(prefetched) == (tensor,)

    expected_consumer_stream = (consumer_stream, t.device("cuda:1"))
    assert calls[0] == ("create_stream", t.device("cuda:1"))
    assert ("stream_enter", producer_stream) in calls
    assert ("create", False) in calls
    assert ("record", producer_stream) in calls
    assert ("stream_exit", producer_stream) in calls
    assert ("wait", expected_consumer_stream) in calls
    assert ("record_stream", expected_consumer_stream) in calls
    assert calls.index(("stream_enter", producer_stream)) < calls.index(
        ("record", producer_stream)
    ) < calls.index(("stream_exit", producer_stream))
    assert calls.index(("record", producer_stream)) < calls.index(
        ("wait", expected_consumer_stream)
    ) < calls.index(("record_stream", expected_consumer_stream))


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
def test_real_cuda_handoff_is_correct_on_each_visible_device_without_global_sync() -> None:
    for device_index in range(t.cuda.device_count()):
        device = t.device("cuda", device_index)
        producer = SFTState()
        consumer = SFTState()
        recorder = TimingRecorder()
        expected_cpu = t.arange(1 << 16, dtype=t.float32).pin_memory()

        def batches(
            producer_state: SFTState = producer,
            source: t.Tensor = expected_cpu,
            target_device: t.device = device,
        ):
            for value in range(4):
                with timed_region("sft/cuda_worker_phase", sync_cuda=False):
                    producer_state.consumed = value + 1
                    gpu = t.empty_like(source, device=target_device)
                    gpu.copy_(source, non_blocking=True)
                    yield_batch = (gpu.clone(),)
                yield yield_batch

        received: list[t.Tensor] = []
        with _sft_step_batch_iterator(
            batches(),
            producer_state=producer,
            consumer_state=consumer,
            batch_count=4,
            prefetch_batches=2,
            cuda_device=device,
            timing_recorder=recorder,
            timing_rank=device_index,
            timing_step=5,
        ) as prefetched:
            for (tensor,) in prefetched:
                received.append(tensor.clone())

        assert consumer.consumed == 4
        assert len(received) == 4
        for tensor in received:
            assert t.equal(tensor.cpu(), expected_cpu)
        assert [row["name"] for row in recorder.rows()] == ["sft/cuda_worker_phase"] * 4
        assert [row["micro_step"] for row in recorder.rows()] == list(range(4))
        assert all(row["rank"] == device_index and row["step"] == 5 for row in recorder.rows())
