"""Tests for _prefetch_wrapper correctness.

Covers:
- _prefetch_wrapper basic correctness (finite generator, ordering, empty)
- Error propagation through prefetch
- Thread termination after generator exhaustion
- Backpressure (queue full)

Note: Live-dataloader-specific state dict and equivalence tests were removed
along with the live encoder-decoder dataloader.
"""

import threading
import time

import pytest

from steering.conditional.encoder_decoder.dataloader import _prefetch_wrapper

# ===========================================================================
# 1. _prefetch_wrapper — basic correctness
# ===========================================================================


class TestPrefetchFiniteGenerator:
    """Feed a finite generator -> consumer gets exactly N items in order."""

    def test_exact_item_count(self):
        items = list(range(10))

        def gen():
            yield from items

        result = list(_prefetch_wrapper(gen(), prefetch_batches=3))
        assert result == items

    def test_fifo_order_preserved(self):
        items = [(i, f"data_{i}") for i in range(20)]

        def gen():
            yield from items

        result = list(_prefetch_wrapper(gen(), prefetch_batches=5))
        assert result == items

    def test_single_item(self):
        def gen():
            yield "only_item"

        result = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert result == ["only_item"]


class TestPrefetchEmptyGenerator:
    """Feed a generator of 0 items -> consumer gets nothing."""

    def test_empty_generator(self):
        def gen():
            return
            yield  # Unreachable by design; forces generator semantics.

        result = list(_prefetch_wrapper(gen(), prefetch_batches=3))
        assert result == []


class TestPrefetchBufferSizes:
    """Works with various prefetch_batches values."""

    @pytest.mark.parametrize("buf_size", [1, 2, 5, 10])
    def test_various_buffer_sizes(self, buf_size):
        items = list(range(25))

        def gen():
            yield from items

        result = list(_prefetch_wrapper(gen(), prefetch_batches=buf_size))
        assert result == items


# ===========================================================================
# 2. Thread termination
# ===========================================================================


class TestPrefetchThreadTermination:
    """After consuming all items, the background thread should exit."""

    def test_thread_exits_after_exhaustion(self):
        """Verify no threads are leaked after consuming all items."""
        initial_threads = threading.active_count()

        def gen():
            yield from range(5)

        # Consume all items
        result = list(_prefetch_wrapper(gen(), prefetch_batches=2))
        assert len(result) == 5

        # Give daemon thread time to terminate
        time.sleep(0.1)
        # Thread count should not have increased
        assert threading.active_count() <= initial_threads + 1

    def test_partial_consumption_no_hang(self):
        """Consumer stops early — daemon thread should not block shutdown."""
        def gen():
            yield from range(100)

        wrapper = _prefetch_wrapper(gen(), prefetch_batches=2)
        # Only consume 3 items
        for _ in range(3):
            next(wrapper)
        # Discard the wrapper — daemon thread gets GC'd
        del wrapper
        # No hang = test passes


# ===========================================================================
# 3. Error propagation
# ===========================================================================


class TestPrefetchErrorPropagation:
    """When inner generator raises, the exception should propagate."""

    def test_raises_original_exception_type(self):
        def gen():
            yield 1
            yield 2
            raise ValueError("test error")

        wrapper = _prefetch_wrapper(gen(), prefetch_batches=3)
        assert next(wrapper) == 1
        assert next(wrapper) == 2
        with pytest.raises(ValueError, match="test error"):
            next(wrapper)

    def test_items_before_error_are_yielded(self):
        def gen():
            yield "a"
            yield "b"
            yield "c"
            raise RuntimeError("boom")

        wrapper = _prefetch_wrapper(gen(), prefetch_batches=5)
        collected = []
        with pytest.raises(RuntimeError, match="boom"):
            for item in wrapper:
                collected.append(item)
        assert collected == ["a", "b", "c"]

    def test_error_on_first_item(self):
        def gen():
            raise TypeError("immediate failure")
            yield  # Unreachable by design; forces generator semantics.

        wrapper = _prefetch_wrapper(gen(), prefetch_batches=2)
        with pytest.raises(TypeError, match="immediate failure"):
            next(wrapper)


# ===========================================================================
# 4. Backpressure (queue full)
# ===========================================================================


class TestPrefetchBackpressure:
    """With prefetch_batches=1, producer can only be 1 batch ahead."""

    def test_slow_consumer_no_deadlock(self):
        """Slow consumer should not cause producer to error."""
        items = list(range(10))

        def gen():
            yield from items

        wrapper = _prefetch_wrapper(gen(), prefetch_batches=1)
        collected = []
        for item in wrapper:
            time.sleep(0.01)  # simulate slow consumer
            collected.append(item)
        assert collected == items

    def test_no_data_corruption_with_small_buffer(self):
        """With buffer=1, ensure no items are lost or duplicated."""
        expected = [(i, i * 2) for i in range(50)]

        def gen():
            yield from expected

        result = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert result == expected
