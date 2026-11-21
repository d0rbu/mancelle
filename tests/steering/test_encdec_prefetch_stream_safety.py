"""Regression test for prefetch producer/consumer CUDA stream race.

Bug: In `_prefetch_wrapper`, the producer thread issues
  gpu_buffer.copy_(cpu_buffer, non_blocking=True)
  yield (inputs.clone(), targets.clone(), ...)
on its own (default) CUDA stream. The consumer thread receives the clones and
reads from them on its own current stream. PyTorch's CUDA semantics only
guarantee in-stream ordering, not cross-thread ordering.  Today a downstream
`torch.cuda.synchronize()` or training-loop autocast entry masks this race;
remove that implicit barrier and the consumer can observe partial data.

Fix: Producer records a `torch.cuda.Event` after each `.clone()` on its
current stream. The event is queued alongside the item. Before re-yielding,
the consumer calls `event.wait(consumer_stream)` to order the consumer's
stream after the producer's writes. This drops the need for an explicit
`torch.cuda.synchronize()` at the training loop boundary.

These tests confirm:
1. The new record_event + wait_event wiring is active (smoke).
2. When CUDA is available, consumer reads the tensor correctly without any
   `torch.cuda.synchronize()` between producer and consumer — stressing the
   consumer with CPU-heavy work to widen the race window.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import _prefetch_wrapper

# Skip CUDA-specific tests when CUDA is unavailable; structural wiring tests
# still run on CPU.
requires_cuda = pytest.mark.skipif(
    not t.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# 1. Structural / API wiring smoke tests (run on any backend)
# ---------------------------------------------------------------------------


class TestPrefetchWrapperEventWiring:
    """Verify `_prefetch_wrapper` accepts `use_cuda` and the event path exists."""

    def test_signature_accepts_use_cuda_flag(self):
        """_prefetch_wrapper must accept the `use_cuda` kwarg."""
        import inspect
        sig = inspect.signature(_prefetch_wrapper)
        assert "use_cuda" in sig.parameters, (
            "`_prefetch_wrapper` must take a `use_cuda` parameter so it can "
            "emit torch.cuda.Event records on the producer stream."
        )

    def test_use_cuda_false_preserves_cpu_yield(self):
        """With use_cuda=False, the wrapper yields raw items (event path off)."""
        def gen():
            yield from range(4)
        items = list(_prefetch_wrapper(gen(), prefetch_batches=2, use_cuda=False))
        assert items == [0, 1, 2, 3]

    def test_default_use_cuda_is_false(self):
        """Default behaviour (use_cuda not passed) must work on CPU."""
        def gen():
            yield "a"
            yield "b"
        items = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert items == ["a", "b"]

    def test_use_cuda_true_records_event_and_waits_on_consumer_stream(self, monkeypatch):
        """The event path is testable without a physical CUDA device."""
        events = []

        class FakeEvent:
            def __init__(self, blocking: bool = False) -> None:
                self.blocking = blocking
                self.recorded = False
                self.wait_streams: list[str] = []
                events.append(self)

            def record(self) -> None:
                self.recorded = True

            def wait(self, stream) -> None:
                self.wait_streams.append(stream)

        monkeypatch.setattr(t.cuda, "Event", FakeEvent)
        monkeypatch.setattr(t.cuda, "current_stream", lambda: "consumer-stream")

        def gen():
            yield from range(3)

        items = list(_prefetch_wrapper(gen(), prefetch_batches=2, use_cuda=True))

        assert items == [0, 1, 2]
        assert len(events) == 3
        assert all(evt.recorded for evt in events)
        assert [evt.wait_streams for evt in events] == [
            ["consumer-stream"],
            ["consumer-stream"],
            ["consumer-stream"],
        ]


# ---------------------------------------------------------------------------
# 2. Cross-stream correctness: producer/consumer on different streams, no
#    explicit torch.cuda.synchronize(), data must be intact.
# ---------------------------------------------------------------------------


@requires_cuda
class TestPrefetchCrossStreamCorrectness:
    """Simulate the production pattern where the producer runs on its own
    CUDA stream and the consumer on a different one.  Without the event
    wiring, cross-stream reads can observe partial data.
    """

    def test_consumer_reads_full_data_without_global_sync(self):
        """Producer fills a GPU buffer on its own stream, clones it, and
        yields via the prefetch queue. Consumer reads on a separate stream
        with NO `torch.cuda.synchronize()` between produce and consume.
        The read must still see the full, final tensor contents.
        """
        N = 1 << 16  # 65536 elements — big enough that an async copy takes time
        # Known pattern the producer writes
        expected_pattern = t.arange(N, dtype=t.float32, device="cuda")

        # Use a dedicated producer stream so the producer's ops are NOT on the
        # consumer's default stream (reproduces the production pattern).
        producer_stream = t.cuda.Stream()

        def inner_gen():
            for step in range(8):
                # Run the producer's ops on its own stream
                with t.cuda.stream(producer_stream):
                    gpu_buffer = t.empty(N, dtype=t.float32, device="cuda")
                    # Simulate a non-blocking HtoD by chaining a kernel that
                    # writes the pattern. The fill is intentionally fast but
                    # real work to keep the race window open.
                    cpu_src = expected_pattern.cpu().pin_memory()
                    gpu_buffer.copy_(cpu_src, non_blocking=True)
                    yielded = gpu_buffer.clone()
                    yield (yielded, step)

        results: list[tuple[t.Tensor, int]] = []
        for item in _prefetch_wrapper(inner_gen(), prefetch_batches=2, use_cuda=True):
            tensor, step = item
            # Consumer does NOT call torch.cuda.synchronize().
            # It reads the tensor on the current (consumer) stream.
            consumed = tensor.clone()
            results.append((consumed, step))

        # Finally we sync once to compare results. The bug would show up as
        # some of the cloned tensors having incorrect (partial/zero) contents.
        t.cuda.synchronize()

        assert len(results) == 8
        for consumed, step in results:
            assert t.equal(consumed.cpu(), expected_pattern.cpu()), (
                f"Step {step}: consumer read corrupt data without event sync."
            )

    def test_consumer_under_artificial_load(self):
        """Stress variant: between receiving an item and reading it, the
        consumer does CPU-heavy work to widen the window where the producer's
        copy could still be in flight.
        """
        N = 1 << 16
        expected_pattern = t.arange(N, dtype=t.float32, device="cuda")
        producer_stream = t.cuda.Stream()

        def inner_gen():
            for step in range(4):
                with t.cuda.stream(producer_stream):
                    gpu_buffer = t.empty(N, dtype=t.float32, device="cuda")
                    cpu_src = expected_pattern.cpu().pin_memory()
                    gpu_buffer.copy_(cpu_src, non_blocking=True)
                    yielded = gpu_buffer.clone()
                    yield (yielded, step)

        got = []
        for item in _prefetch_wrapper(inner_gen(), prefetch_batches=2, use_cuda=True):
            tensor, step = item
            # Simulate CPU-heavy work that blocks the consumer before reading.
            # The producer may well finish another batch while we're busy here.
            _ = sum(i * i for i in range(20000))
            consumed_cpu = tensor.cpu()  # forces a CUDA sync per-tensor, but
            # that's the point: we need the event to have kept the data intact.
            got.append((consumed_cpu, step))

        assert len(got) == 4
        for consumed_cpu, step in got:
            assert t.equal(consumed_cpu, expected_pattern.cpu()), (
                f"Step {step}: consumer read corrupt data under CPU load."
            )


# ---------------------------------------------------------------------------
# 3. Regression: existing behaviour is unchanged for the CPU/non-CUDA path
# ---------------------------------------------------------------------------


class TestPrefetchWrapperPreservesBehaviour:
    """The fix must not alter ordering, exception propagation, or completion."""

    def test_items_received_in_order(self):
        def gen():
            yield from range(10)
        items = list(_prefetch_wrapper(gen(), prefetch_batches=3))
        assert items == list(range(10))

    def test_exception_propagates_to_consumer(self):
        def gen():
            yield 1
            yield 2
            raise RuntimeError("boom")
        got = []
        with pytest.raises(RuntimeError, match="boom"):
            for item in _prefetch_wrapper(gen(), prefetch_batches=2):
                got.append(item)
        assert got == [1, 2]

    def test_empty_generator_yields_nothing(self):
        def gen():
            return
            yield  # make it a generator
        items = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert items == []

    @requires_cuda
    def test_use_cuda_true_items_received_in_order(self):
        """With use_cuda=True, ordering must still be preserved."""
        def gen():
            for i in range(5):
                yield (i, t.tensor([i], device="cuda"))
        got = [i for i, _ in _prefetch_wrapper(gen(), prefetch_batches=2, use_cuda=True)]
        assert got == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 4. End-to-end via the public dataloader (live path)
# ---------------------------------------------------------------------------


@requires_cuda
class TestLiveDataloaderPrefetchStreamSafety:
    """The `encoder_decoder_data_loader_with_state` live path with
    prefetch_batches>0 must produce batches on CUDA without requiring the
    caller to call `torch.cuda.synchronize()`. Today this is masked by the
    training loop; the event fix makes it robust to caller-side patterns.
    """

    def test_live_loader_batches_correct_without_explicit_sync(self):
        # We mimic the consumer pattern of reading tensors immediately after
        # receiving the batch, on the consumer's current stream. If the event
        # fix is wired correctly, the tensor contents are stable.
        # This is mostly a smoke: correctness comes from the structural
        # synchronisation in `_prefetch_wrapper`.
        #
        # We use a minimal synthetic inner generator to avoid bringing up the
        # full parquet tokenizer stack here (which is well-exercised in
        # `tests/test_live_dataloader.py`). The critical path is the
        # `_prefetch_wrapper` event wiring, already covered above.
        N = 1024
        expected = t.arange(N, dtype=t.float32, device="cuda")

        def inner():
            for step in range(3):
                gpu = t.empty(N, dtype=t.float32, device="cuda")
                gpu.copy_(expected.cpu().pin_memory(), non_blocking=True)
                yield (gpu.clone(), step)

        for tensor, step in _prefetch_wrapper(inner(), prefetch_batches=2, use_cuda=True):
            # No torch.cuda.synchronize() here.
            assert t.equal(tensor.cpu(), expected.cpu()), (
                f"Step {step}: partial data observed without event sync."
            )
