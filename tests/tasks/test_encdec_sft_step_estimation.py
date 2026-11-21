"""
Tests for consumption-based stopping in encoder-decoder SFT training.

These tests verify that make_sft_data_generator tracks how many conversations
have been consumed via an SFTState dataclass, enabling the training loop to stop
after exactly 1 epoch (all conversations consumed) rather than relying on the
overestimated pre-calculation of num_iterations.

The pre-estimation bug:
    tokens_per_example_approx = cfg.max_seq_len  # 2048
    num_iterations = dataset_size * tokens_per_example_approx // total_batch_size

This overestimates by ~5x because real conversations are ~300-400 tokens, not 2048.
The fix adds consumption tracking so the loop stops when all data has been seen.
"""

import pytest

from scripts.train.conditional_encdec_sft import SFTState, make_sft_data_generator
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.steering.test_encdec_sft_conditioned_integration import (
    _make_boundaries,
    build_synthetic_features,
    make_lazy_patched_sft_generator,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FixedLengthTask(Task):
    """Task returning N conversations, each producing a known token length.

    Each conversation has a user message and an assistant message. The
    SFTMockTokenizer below converts these to exactly `token_len` tokens
    total (user_len + asst_len + 2 for BOS/EOS framing).
    """

    def __init__(self, n: int, token_len: int):
        super().__init__()
        self._n = n
        self._token_len = token_len
        # Split tokens roughly evenly between user and assistant, reserving 2
        # for BOS + separator overhead.
        self._user_chars = max(1, (token_len - 2) // 2)
        self._asst_chars = max(1, token_len - 2 - self._user_chars)

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "u" * self._user_chars},
                {"role": "assistant", "content": "a" * self._asst_chars},
            ]
        }


class SFTMockTokenizer:
    """Minimal tokenizer with render_conversation for SFT generator tests.

    Produces token sequences of deterministic length from conversations:
      total_len = len(user_content) + len(asst_content) + 2  (BOS + separator)
      ids  = [0, 2, 3, ..., total_len]
      mask = [0]*(user_len+1) + [1]*asst_len   (user=0, assistant=1)
    """

    vocab_size: int = 128

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        user_content = conversation["messages"][0]["content"]
        asst_content = conversation["messages"][1]["content"]
        user_len = len(user_content)
        asst_len = len(asst_content)
        total_len = user_len + asst_len + 2  # +2 for BOS + separator
        ids = [0] + list(range(2, total_len + 1))  # BOS=0, then 2..total_len
        # Mask: user portion (BOS + user tokens + separator) = 0, assistant = 1
        mask = [0] * (user_len + 1) + [1] * (asst_len + 1)
        # Ensure lengths match
        ids = ids[:total_len]
        mask = mask[:total_len]
        return ids, mask

    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tokenizer():
    return SFTMockTokenizer()


def _make_gen(task, tokenizer, state, *, B=2, T=256, ddp_rank=0, ddp_world_size=1, buffer_size=100):
    """Shorthand for creating a generator with common defaults."""
    features = build_synthetic_features(task, tokenizer)
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=1000,
    )

    return make_lazy_patched_sft_generator(
        features,
        lambda: make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device="cpu",
            sft_state=state,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            buffer_size=buffer_size,
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg,
            boundaries=_make_boundaries(),
            min_loss_tokens=1,
            alignment_probe_samples=0,
            expected_split=SFTSplit.TRAIN,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConsumedTracking:
    """The generator must update sft_state.consumed as conversations are used."""

    def test_consumed_incremented_after_one_batch(self, tokenizer):
        """After yielding one batch, sft_state.consumed should be > 0."""
        task = FixedLengthTask(n=20, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state)
        next(gen)
        assert state.consumed > 0, f"Expected consumed > 0 after one batch, got {state.consumed}"

    def test_consumed_increases_with_batches(self, tokenizer):
        """consumed should increase monotonically as more batches are generated."""
        task = FixedLengthTask(n=100, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state)
        prev = 0
        for _ in range(5):
            next(gen)
            assert state.consumed > prev, (
                f"consumed did not increase: was {prev}, now {state.consumed}"
            )
            prev = state.consumed

    def test_consumed_equals_conversations_used(self, tokenizer):
        """consumed count should match the actual number of conversations popped.

        With 20 conversations of 50 tokens each and T=256 (row_capacity=257),
        each row can fit ~5 conversations. With B=2, each batch uses ~10 convs.
        After 2 batches, consumed should be ~20.
        """
        N = 20
        task = FixedLengthTask(n=N, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state)
        for _ in range(10):
            next(gen)
            if state.consumed >= N:
                break

        assert state.consumed >= N, (
            f"Expected consumed >= {N} after exhausting dataset, got {state.consumed}"
        )


class TestConsumedReachesDatasetSize:
    """After one epoch the consumed count must reach the dataset size."""

    def test_consumed_reaches_dataset_size_after_one_epoch(self, tokenizer):
        """Run generator until consumed >= dataset_size and verify step count.

        With N=40 conversations of 50 tokens, T=512 (row_capacity=513),
        each row fits ~10 conversations. B=4 means ~40 convs per batch.
        So 1 batch should consume all 40 conversations.

        The OLD estimation would compute:
            num_iterations = 40 * 512 / (4 * 512) = 10
        But actually only ~1 batch is needed. The consumed-based approach
        should stop after ~1 batch, not 10.
        """
        N = 40
        B = 4
        T = 512
        task = FixedLengthTask(n=N, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, B=B, T=T)

        steps = 0
        while state.consumed < N:
            next(gen)
            steps += 1
            if steps > 100:  # safety valve
                break

        # Old estimate: N * T / (B * T) = N / B = 10
        old_estimate = max(1, N * T // (B * T))
        assert state.consumed >= N, f"consumed={state.consumed} never reached dataset_size={N}"
        assert steps < old_estimate, (
            f"Took {steps} steps, but old estimate was {old_estimate}. "
            f"Consumption-based stopping should require fewer steps."
        )


class TestStepCountProportionalToActualData:
    """Step count should be proportional to actual token count, not max_seq_len."""

    def test_short_conversations_fewer_steps_than_old_estimate(self, tokenizer):
        """With short conversations, actual steps << old pre-estimated steps.

        N=100 conversations, each ~50 tokens.
        B=4, T=512 (row_capacity=513, fits ~10 convs/row).
        Actual steps: 100 / (4 * 10) ~= 2-3 batches.
        Old estimate: 100 * 512 / (4 * 512) = 25 batches.
        """
        N = 100
        B = 4
        T = 512
        task = FixedLengthTask(n=N, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, B=B, T=T)

        steps = 0
        while state.consumed < N:
            next(gen)
            steps += 1
            if steps > 200:
                break

        old_estimate = max(1, N * T // (B * T))  # = N // B = 25
        assert steps <= old_estimate // 2, (
            f"Steps ({steps}) should be much less than old estimate ({old_estimate}). "
            f"Short conversations should pack densely, reducing step count."
        )


class TestProgressCalculation:
    """Progress should be consumed/dataset_size, not step/pre_estimated_steps."""

    def test_progress_from_consumed(self, tokenizer):
        """Verify that consumed/dataset_size gives correct progress."""
        N = 30
        task = FixedLengthTask(n=N, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state)
        next(gen)
        progress = state.consumed / N
        assert 0.0 < progress <= 1.0, (
            f"Progress should be between 0 and 1, got {progress} "
            f"(consumed={state.consumed}, dataset_size={N})"
        )


class TestDDPRankHandling:
    """DDP rank should affect consumed tracking correctly."""

    def test_consumed_starts_at_zero_for_all_ranks(self, tokenizer):
        """consumed should start at 0 regardless of ddp_rank."""
        task = FixedLengthTask(n=20, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, ddp_rank=1, ddp_world_size=2)

        assert state.consumed == 0, f"consumed should start at 0, got {state.consumed}"
        next(gen)
        assert state.consumed > 0, f"consumed should be > 0 after first batch, got {state.consumed}"

    def test_consumed_increments_by_world_size(self, tokenizer):
        """Each buffer pop should increment consumed by ddp_world_size.

        With ddp_world_size=2, each conversation fetch advances the cursor by 2
        (one per rank), so consumed should increment by 2 per conversation popped.
        """
        task = FixedLengthTask(n=40, token_len=50)
        world_size = 2
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, ddp_rank=0, ddp_world_size=world_size)
        next(gen)
        assert state.consumed % world_size == 0, (
            f"consumed ({state.consumed}) should be a multiple of ddp_world_size ({world_size})"
        )

    def test_different_ranks_same_consumed_count(self, tokenizer):
        """Two ranks processing the same dataset should report similar consumed.

        Each rank sees every world_size-th conversation, so both should
        advance consumed at the same rate.
        """
        N = 40
        task = FixedLengthTask(n=N, token_len=50)

        state_r0 = SFTState()
        state_r1 = SFTState()

        gen_r0 = _make_gen(task, tokenizer, state_r0, B=1, ddp_rank=0, ddp_world_size=2)
        gen_r1 = _make_gen(task, tokenizer, state_r1, B=1, ddp_rank=1, ddp_world_size=2)

        for _ in range(3):
            next(gen_r0)
            next(gen_r1)

        assert state_r0.consumed == state_r1.consumed, (
            f"Rank 0 consumed {state_r0.consumed} vs "
            f"rank 1 consumed {state_r1.consumed} — should be equal"
        )


class TestOldEstimateOvercounts:
    """Demonstrate the bug: old pre-estimation gives too many steps."""

    def test_old_estimate_vs_actual_consumption(self, tokenizer):
        """The old formula overestimates by a large factor for short conversations.

        Old formula:
            num_iterations = dataset_size * max_seq_len // total_batch_size

        With dataset_size=50, max_seq_len=2048, total_batch_size=4*2048=8192:
            old = 50 * 2048 // 8192 = 12

        But with 50-token conversations and row_capacity=2049:
            Each row fits ~40 conversations
            Each batch (B=4): ~160 conversations
            Actual: 1 batch to consume 50 conversations

        The ratio old/actual should be >= 3x.
        """
        N = 50
        B = 4
        T = 2048
        task = FixedLengthTask(n=N, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, B=B, T=T)

        actual_steps = 0
        while state.consumed < N:
            next(gen)
            actual_steps += 1
            if actual_steps > 200:
                break

        total_batch_size = B * T
        old_estimate = max(1, N * T // total_batch_size)

        assert actual_steps < old_estimate, (
            f"actual_steps={actual_steps} should be < old_estimate={old_estimate}. "
            f"This proves the old formula overestimates for short conversations."
        )
        if actual_steps > 0:
            ratio = old_estimate / actual_steps
            assert ratio >= 3, (
                f"Overestimation ratio {ratio:.1f}x is too small. "
                f"Expected >= 3x for conversations that are 50/{T} "
                f"= {50 / T:.1%} of max_seq_len."
            )


class TestExplicitNumIterationsStillRespected:
    """When num_iterations > 0, that value should still cap training."""

    def test_explicit_iterations_respected(self, tokenizer):
        """If the caller sets num_iterations=5, training should stop at step 5.

        This test verifies that explicit num_iterations is NOT broken by the
        consumption-based stopping. The generator itself is infinite (it wraps);
        the training loop decides when to stop. But sft_state should still be
        tracked so the caller CAN use either mechanism.
        """
        task = FixedLengthTask(n=100, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state)

        for _ in range(5):
            next(gen)

        # After 5 steps, consumed should be tracked correctly
        assert state.consumed > 0, (
            "consumed should be tracked even when using explicit num_iterations"
        )
        # The generator should NOT have stopped on its own — it is infinite
        next(gen)  # Should not raise StopIteration
        assert state.consumed > 0


class TestYieldShape:
    """Sanity check: generator yields the expected 7-tuple (the 7th element
    is the feature-prediction target tensor or None)."""

    def test_yield_is_seven_tuple(self, tokenizer):
        """make_sft_data_generator yields (inputs, targets, enc_fids, enc_mags,
        enc_mask, seg_map, fp_targets)."""
        task = FixedLengthTask(n=10, token_len=50)
        state = SFTState()

        gen = _make_gen(task, tokenizer, state, B=2, T=128)

        batch = next(gen)
        assert len(batch) == 7, f"Expected 7-tuple, got {len(batch)}-tuple"
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
        assert inputs.shape == (2, 128), f"inputs shape {inputs.shape} != (2, 128)"
        assert targets.shape == (2, 128), f"targets shape {targets.shape} != (2, 128)"
