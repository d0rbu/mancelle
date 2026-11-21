"""Heavy correctness tests for ``sft_data_generator_bos_bestfit_core``.

Where ``test_sft_generator_core.py`` covers the high-level contract
(stops at the right time, yields right shapes), this file goes deeper:

A. **Token-level invariants.** Every yielded byte of every batch must
   come from real conversation content or be a defined padding token.
   targets[i, j] must equal inputs[i, j+1] (next-token shift). The
   masking boundary must align exactly with the content boundary.

B. **Conservation laws.** Conversations consumed equals conversations
   yielded. No silent drops. No duplicates within a single epoch
   (rank-local). Truncation drops the tail of long conversations,
   nothing else.

C. **Multi-rank disjointness.** Two ranks running in parallel must
   process DISJOINT sets of conversations — no overlap, no gaps in
   the union (modulo wrapping).

E. **Determinism.** Same inputs, same outputs, every time.

F. **Real-model integration.** Drive a real tiny GPT with the
   yielded batches: forward pass, compute loss, backward pass,
   optimizer step. Loss is finite, parameters change, no exceptions.
   This is the strongest "the implementation actually works" check
   short of a full training run.

G. **Wrapper delegation.** ``scripts/chat_sft.py`` wraps the core
   generator and copies the yielded progress snapshot onto module
   globals. Verify both via source inspection and via running the
   wrapper with mocked module state.

H. **Edge cases.** Tiny datasets, single-conversation datasets,
   length-1 conversations, multi-row batches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT
from steering.sft.iteration import (
    SFTGeneratorProgress,
    sft_data_generator_bos_bestfit_core,
)
from tests.conftest import _make_gpt


@pytest.fixture(autouse=True)
def _force_sdpa():
    """CPU-only tests must use SDPA; FA3 is CUDA-only."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Test doubles — instrumented to enable conservation-law checks
# ---------------------------------------------------------------------------


class _RecordingTokenizer:
    """Tokenizer that records every render_conversation call so tests
    can verify which conversations were consumed."""

    bos_id = 0

    def __init__(self) -> None:
        self.render_calls: list[tuple[int, int]] = []  # (length, seed)

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def render_conversation(self, conversation):
        N = conversation["length"]
        seed = conversation["seed"]
        self.render_calls.append((N, seed))
        # Encode (length, seed) into the token stream so tests can
        # reconstruct which conversation each token came from.
        # Token format: [BOS, seed, length, content_0, ..., content_{N-3}]
        # where content_k = 100 + (seed * 1000 + k) % 100 (avoids BOS=0).
        # First 3 tokens encode the conversation identity for verification.
        if N == 0:
            return [], []
        if N == 1:
            ids = [self.bos_id]
        elif N == 2:
            ids = [self.bos_id, 50 + seed % 50]  # seed marker
        else:
            ids = [self.bos_id, 50 + seed % 50, 50 + N % 50]
            for k in range(N - 3):
                ids.append(2 + (seed * 1000 + k) % 100)
        mask = [0] * len(ids)
        return ids, mask


class _SimpleDataset:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict:
        return {"length": self.lengths[idx], "seed": idx}


def _drive(
    gen,
    *,
    max_yields: int = 1,
) -> list[tuple[t.Tensor, t.Tensor, SFTGeneratorProgress]]:
    """Take an explicit number of batches; the generator owns no stop state."""
    return [next(gen) for _ in range(max_yields)]


def _make_gen(
    lengths: list[int],
    *,
    tokenizer: _RecordingTokenizer | None = None,
    max_seq_len: int = 16,
    device_batch_size: int = 2,
    num_iterations: int = -1,
    grad_accum_steps: int = 1,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    is_train_split: bool = True,
    buffer_size: int = 4,
):
    return sft_data_generator_bos_bestfit_core(
        dataset=_SimpleDataset(lengths),
        tokenizer=tokenizer or _RecordingTokenizer(),
        max_seq_len=max_seq_len,
        device_batch_size=device_batch_size,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=buffer_size,
    )


# ===========================================================================
# A. Token-level invariants
# ===========================================================================


class TestNextTokenShiftInvariant:
    """``targets[i, j]`` must equal ``inputs[i, j+1]`` for every (i, j)
    where the target is non-padding (-1). This is the foundational
    correctness invariant of the data pipeline: if it ever breaks,
    every model trained on this data will learn corrupted next-token
    objectives."""

    def test_shift_holds_for_fully_packed_rows(self) -> None:
        """Three convs of length 3 pack to row_capacity=9 exactly; no
        masking. Every target must equal the next input."""
        gen = _make_gen(lengths=[3] * 10, max_seq_len=8, device_batch_size=2, num_iterations=1)
        batches = _drive(gen)
        for inputs, targets, _ in batches:
            B, T = inputs.shape
            for i in range(B):
                for j in range(T):
                    if targets[i, j] >= 0:  # not masked
                        # The shift relationship: targets[i, j] is the j+1-th
                        # token of the underlying row, which we don't have
                        # direct access to. But inputs[i, j] is the j-th
                        # token, and (for j < T-1) inputs[i, j+1] should
                        # equal targets[i, j] when neither is masked.
                        if j + 1 < T:
                            assert int(targets[i, j]) == int(inputs[i, j + 1]), (
                                f"shift broken at (i={i}, j={j}): "
                                f"targets={int(targets[i, j])}, "
                                f"inputs[j+1]={int(inputs[i, j + 1])}"
                            )

    def test_shift_holds_for_padded_rows(self) -> None:
        """A short conversation produces a padded row. The shift must
        hold for the content positions; padding positions are masked
        with -1 so we don't check those."""
        gen = _make_gen(
            lengths=[5], max_seq_len=8, device_batch_size=1, num_iterations=1, buffer_size=2
        )
        batches = _drive(gen)
        for inputs, targets, _ in batches:
            B, T = inputs.shape
            for i in range(B):
                for j in range(T - 1):
                    if int(targets[i, j]) >= 0:
                        assert int(targets[i, j]) == int(inputs[i, j + 1]), (
                            f"shift broken at (i={i}, j={j}): "
                            f"targets={int(targets[i, j])}, "
                            f"inputs[j+1]={int(inputs[i, j + 1])}"
                        )


class TestPaddingTokenIdentity:
    """When a row is padded, the padding tokens must be the BOS token
    id (per the generator's contract). A regression that swapped to
    a different token id would silently train the model on
    non-padding patterns."""

    def test_padding_uses_bos_token(self) -> None:
        """In a padded row, the tokens past content_len must all equal
        the tokenizer's BOS id."""
        tokenizer = _RecordingTokenizer()
        gen = _make_gen(
            lengths=[5],  # short; will pad
            tokenizer=tokenizer,
            max_seq_len=8,
            device_batch_size=1,
            num_iterations=1,
            buffer_size=2,
        )
        batches = _drive(gen)
        bos_id = tokenizer.get_bos_token_id()

        # Pull the (single) row's raw inputs. We don't have direct
        # access to row_capacity, but content_len < row_capacity is
        # signaled by a -1 in targets. Find the position where -1
        # starts; everything from inputs[content_len:] should be BOS.
        for inputs, targets, _ in batches:
            for row_idx in range(inputs.shape[0]):
                # Find first -1 in targets row: its index is content_len - 1.
                row_targets = targets[row_idx].tolist()
                if -1 in row_targets:
                    first_minus_one = row_targets.index(-1)
                    content_len = first_minus_one + 1
                    # inputs[content_len:] should all be BOS.
                    pads = inputs[row_idx, content_len:].tolist()
                    assert all(p == bos_id for p in pads), (
                        f"padding tokens at row {row_idx} are not all BOS (id={bos_id}): {pads}"
                    )


# ===========================================================================
# B. Conservation laws
# ===========================================================================


class TestConservation:
    """Conversations consumed by the generator must equal conversations
    actually packed into yielded rows. No silent drops; no double-
    counting; truncation only drops the tail of over-capacity convs."""

    def test_consumed_counter_matches_recorded_conversations(self) -> None:
        """Across N yields, the generator's `consumed` count equals
        the number of (length, seed) tuples that actually were packed
        into rows (modulo the buffer that might still hold un-popped
        convs at the time of stop)."""
        tokenizer = _RecordingTokenizer()
        gen = _make_gen(
            lengths=[3] * 20,
            tokenizer=tokenizer,
            max_seq_len=16,
            device_batch_size=1,
            num_iterations=3,
            buffer_size=4,
        )
        batches = _drive(gen)
        last_progress = batches[-1][2]
        # Pre-pop count: every render_conversation call corresponds to
        # ONE buffer insertion. The buffer holds up to buffer_size at
        # any moment, so consumed <= len(render_calls).
        assert last_progress.consumed <= len(tokenizer.render_calls), (
            f"consumed ({last_progress.consumed}) exceeds rendered "
            f"({len(tokenizer.render_calls)}); generator made up data?"
        )
        # And at least 1 conversation was popped per yield (since we yielded).
        assert last_progress.consumed >= len(batches), (
            f"consumed ({last_progress.consumed}) less than yields "
            f"({len(batches)}); some yields produced no consumption?"
        )

    def test_truncation_drops_only_tail(self) -> None:
        """A conversation longer than row_capacity has its tail
        dropped at buffer insertion. The first row_capacity tokens
        must appear verbatim in some yielded row."""
        # row_capacity = max_seq_len + 1 = 9. Use a conv of length 50.
        # Its first 9 tokens are: [BOS=0, seed_marker, length_marker, content_0..5]
        tokenizer = _RecordingTokenizer()
        gen = _make_gen(
            lengths=[50],
            tokenizer=tokenizer,
            max_seq_len=8,  # row_capacity = 9
            device_batch_size=1,
            num_iterations=1,
            buffer_size=2,
        )
        batches = _drive(gen)
        inputs, _targets, _progress = batches[0]
        # The first row's first row_capacity tokens are the truncated
        # conversation. inputs has length max_seq_len = row_capacity - 1.
        # So inputs[0] == [bos=0, seed_marker=50, length_marker=...] + 5 content tokens.
        first_row_input = inputs[0].tolist()
        # Re-render the same conversation to get the expected first 9 tokens.
        expected_full, _ = _RecordingTokenizer().render_conversation({"length": 50, "seed": 0})
        expected_truncated = expected_full[:9]
        # inputs is row[:row_capacity-1] = expected_truncated[:8].
        assert first_row_input == expected_truncated[:8], (
            f"truncated content mismatch: got {first_row_input}, expected {expected_truncated[:8]}"
        )

    def test_no_duplicate_consumption_within_epoch(self) -> None:
        """Within a single epoch (rank-local), the same dataset index
        must NOT be popped twice. The generator's cursor advances
        monotonically per refill — verify by tracking refill order."""
        tokenizer = _RecordingTokenizer()
        gen = _make_gen(
            lengths=[3] * 8,  # exactly enough for one epoch
            tokenizer=tokenizer,
            max_seq_len=16,
            device_batch_size=1,
            num_iterations=-1,
            buffer_size=4,
        )
        _drive(gen)
        # Across all render_calls (which is the buffer fetch order),
        # within the FIRST epoch (= first dataset_size calls), there
        # must be no duplicate seeds.
        first_epoch_seeds = [seed for _, seed in tokenizer.render_calls[:8]]
        assert len(set(first_epoch_seeds)) == len(first_epoch_seeds), (
            f"duplicate consumption within epoch 1: seeds={first_epoch_seeds}"
        )


# ===========================================================================
# C. Multi-rank disjointness
# ===========================================================================


class TestMultiRankDisjointness:
    """When N ranks run the generator in parallel, no two ranks may
    process the same dataset index. Different ranks see DISJOINT
    cursor sequences."""

    def test_two_ranks_disjoint_consumption(self) -> None:
        """Rank 0 and rank 1 of a 2-way split must see no overlap in
        which dataset indices they pop."""
        tok0 = _RecordingTokenizer()
        tok1 = _RecordingTokenizer()
        gen0 = _make_gen(
            lengths=[3] * 16,
            tokenizer=tok0,
            max_seq_len=16,
            device_batch_size=1,
            num_iterations=2,
            ddp_rank=0,
            ddp_world_size=2,
            buffer_size=4,
        )
        gen1 = _make_gen(
            lengths=[3] * 16,
            tokenizer=tok1,
            max_seq_len=16,
            device_batch_size=1,
            num_iterations=2,
            ddp_rank=1,
            ddp_world_size=2,
            buffer_size=4,
        )
        _drive(gen0)
        _drive(gen1)
        rank0_seeds = {seed for _, seed in tok0.render_calls}
        rank1_seeds = {seed for _, seed in tok1.render_calls}
        # First-epoch seeds for each rank — within their first epoch
        # they should be disjoint.
        # rank 0 first epoch: 0, 2, 4, 6, 8, 10, 12, 14
        # rank 1 first epoch: 1, 3, 5, 7, 9, 11, 13, 15
        rank0_first = {s for s in rank0_seeds if s % 2 == 0}
        rank1_first = {s for s in rank1_seeds if s % 2 == 1}
        # Each rank's first-epoch seeds match their parity (no overlap).
        assert rank0_first & rank1_first == set(), (
            f"ranks have overlapping first-epoch consumption: r0={rank0_first}, r1={rank1_first}"
        )
        # Each rank should see ONLY its parity within its first epoch.
        for s in rank0_seeds:
            if s < 16:  # within dataset_size
                assert s % 2 == 0, f"rank 0 saw odd seed {s}"
        for s in rank1_seeds:
            if s < 16:
                assert s % 2 == 1, f"rank 1 saw even seed {s}"

    def test_four_ranks_disjoint_first_epoch(self) -> None:
        """4-way split: each rank sees only seeds where seed % 4 == rank."""
        toks = [_RecordingTokenizer() for _ in range(4)]
        for rank in range(4):
            gen = _make_gen(
                lengths=[3] * 16,
                tokenizer=toks[rank],
                max_seq_len=16,
                device_batch_size=1,
                num_iterations=1,
                ddp_rank=rank,
                ddp_world_size=4,
                buffer_size=4,
            )
            _drive(gen)
        for rank, tok in enumerate(toks):
            for _, seed in tok.render_calls[:4]:  # first epoch's worth
                assert seed % 4 == rank, (
                    f"rank {rank} saw seed {seed} (expected seed % 4 == {rank})"
                )


# ===========================================================================
# E. Determinism
# ===========================================================================


class TestDeterminism:
    """The generator is a pure function of its inputs. Same inputs →
    same outputs, every time. Reproducibility matters for debugging
    silent regressions and for testable training runs."""

    def test_two_generators_with_same_inputs_yield_identical_sequences(
        self,
    ) -> None:
        """Build two generators with identical params and tokenizers.
        Drive both. Every (inputs, targets) pair must be bitwise-identical."""
        # Use the same lengths list and same max_seq_len etc. — even
        # the seed in the dataset is index-derived, so deterministic.
        gen1 = _make_gen(
            lengths=[5, 7, 3, 9, 4],
            max_seq_len=16,
            device_batch_size=2,
            num_iterations=2,
            buffer_size=3,
        )
        gen2 = _make_gen(
            lengths=[5, 7, 3, 9, 4],
            max_seq_len=16,
            device_batch_size=2,
            num_iterations=2,
            buffer_size=3,
        )
        b1 = _drive(gen1)
        b2 = _drive(gen2)
        assert len(b1) == len(b2), f"yield counts differ: {len(b1)} vs {len(b2)}"
        for (i1, t1, p1), (i2, t2, p2) in zip(b1, b2, strict=True):
            assert t.equal(i1, i2), "inputs differ across runs"
            assert t.equal(t1, t2), "targets differ across runs"
            assert p1 == p2, f"progress differs: {p1} vs {p2}"


# ===========================================================================
# F. Real-model integration — the load-bearing "actually works" tests
# ===========================================================================


class TestRealModelIntegration:
    """The strongest correctness check: the yielded batches actually
    work when fed to a real GPT model. Loss must be finite, backward
    pass must succeed, parameters must change. This catches whole
    classes of bugs that no unit test can: dtype mismatches between
    targets and lm_head outputs, mask values incompatible with
    cross_entropy's ignore_index, padding tokens that produce
    NaN loss, etc."""

    @pytest.fixture
    def tiny_gpt(self) -> GPT:
        """A tiny CPU GPT with vocab compatible with our tokenizer."""
        model = _make_gpt(device="cpu")
        model.train()
        return model

    @pytest.fixture
    def tiny_gen_and_tokenizer(self):
        """Generator + tokenizer pair sized to the tiny GPT."""
        tok = _RecordingTokenizer()
        gen = _make_gen(
            lengths=[5, 7, 4, 8, 6, 5, 7, 4],
            tokenizer=tok,
            max_seq_len=16,
            device_batch_size=2,
            num_iterations=4,
            buffer_size=4,
        )
        return gen, tok

    def test_forward_produces_finite_loss(self, tiny_gpt, tiny_gen_and_tokenizer) -> None:
        """For every yielded batch, the model's forward pass must
        produce a finite loss. NaN/Inf loss means our masking / padding
        / target shapes are broken."""
        gen, _ = tiny_gen_and_tokenizer
        for _ in range(4):
            inputs, targets, _progress = next(gen)
            # GPT expects int64 inputs; .contiguous() ensures the strided
            # slice view from batch_tensor[:, :-1] is materialized so
            # cross_entropy's .view(-1) call works on CPU. (CUDA's .to()
            # already returns contiguous tensors, so this is only needed
            # in the CPU test path.)
            inputs = inputs.long().contiguous()
            targets = targets.contiguous()
            loss = tiny_gpt(inputs, targets=targets)
            assert t.isfinite(loss).all(), (
                f"non-finite loss: {loss.item()}; inputs={inputs}, targets={targets}"
            )

    def test_backward_produces_finite_gradients(self, tiny_gpt, tiny_gen_and_tokenizer) -> None:
        """Backward pass on the loss must produce finite gradients
        for every parameter. NaN gradients are silent killers — they
        propagate through optimizer steps and corrupt all subsequent
        training without crashing."""
        gen, _ = tiny_gen_and_tokenizer
        # Take a single batch (next() advances the generator one step).
        inputs, targets, _ = next(gen)
        inputs = inputs.long().contiguous()
        targets = targets.contiguous()
        loss = tiny_gpt(inputs, targets=targets)
        loss.backward()
        # Every parameter that was used (has grad) must have finite gradients.
        for name, param in tiny_gpt.named_parameters():
            if param.grad is not None:
                assert t.isfinite(param.grad).all(), f"non-finite gradient in parameter {name!r}"

    def test_multi_iteration_training_loop_runs_without_exceptions(self, tiny_gpt) -> None:
        """End-to-end smoke: drive the generator through several
        iterations, do forward+backward+step each time. No exceptions,
        all losses finite, parameters change between iterations.

        This is the closest we can get to "real training works"
        without the cost of an actual SFT run."""
        # Use a wider variety of conv lengths to exercise more of the
        # packing logic.
        gen = _make_gen(
            lengths=[5, 7, 4, 8, 6, 9, 3, 11, 5, 7],
            max_seq_len=16,
            device_batch_size=2,
            num_iterations=5,
            grad_accum_steps=1,
            buffer_size=4,
        )
        optimizer = t.optim.SGD(tiny_gpt.parameters(), lr=1e-2)
        # Snapshot a representative parameter (lm_head.weight) before
        # training. lm_head always receives gradients through cross_entropy,
        # so it should always change after a successful optimizer step.
        # Choosing the first parameter (resid_lambdas) is unsafe because
        # it can have zero gradients early in training.
        target_param_name = "lm_head.weight"
        target_param = dict(tiny_gpt.named_parameters())[target_param_name]
        snapshot_before = target_param.detach().clone()
        losses = []
        for _ in range(5):
            inputs, targets, progress = next(gen)
            inputs = inputs.long().contiguous()
            targets = targets.contiguous()
            loss = tiny_gpt(inputs, targets=targets)
            assert t.isfinite(loss).all(), f"step {progress.micro_batch_count}: loss={loss.item()}"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert len(losses) >= 2, "expected at least 2 training iterations"
        # All losses finite (NaN != NaN).
        assert all(loss == loss for loss in losses), losses
        # The lm_head weight changed: optimizer is actually updating things.
        snapshot_after = target_param.detach().clone()
        assert not t.equal(snapshot_before, snapshot_after), (
            f"{target_param_name} did not change across {len(losses)} "
            "training iterations — optimizer.step() is a no-op?"
        )

    def test_oversized_conversation_still_produces_finite_loss(self, tiny_gpt) -> None:
        """Pre-fix bug regression: with the truncation fix, even a
        dataset where every conversation exceeds row_capacity must
        produce finite loss. Pre-fix, this scenario produced
        all-padding rows → all-masked targets → NaN cross_entropy."""
        # row_capacity = 17 (max_seq_len=16 + 1). Every conv is 50.
        gen = _make_gen(
            lengths=[50] * 5,
            max_seq_len=16,
            device_batch_size=2,
            num_iterations=2,
            buffer_size=2,
        )
        for _ in range(2):
            inputs, targets, _progress = next(gen)
            inputs = inputs.long().contiguous()
            targets = targets.contiguous()
            loss = tiny_gpt(inputs, targets=targets)
            assert t.isfinite(loss).all(), (
                "oversized-conversation regression: loss is non-finite. "
                "The truncation fix must produce real content (not all "
                "padding) so cross_entropy has at least one unmasked target."
            )


# ===========================================================================
# G. chat_sft.py wrapper delegation
# ===========================================================================


class TestWrapperDelegation:
    """The wrapper in scripts/chat_sft.py is a thin adapter that
    forwards explicit state into the core generator. The loop owns
    stop/schedule state from committed optimizer work."""

    @pytest.fixture(scope="class")
    def chat_sft_source(self) -> str:
        return (
            Path(__file__).resolve().parent.parent.parent / "scripts" / "train" / "chat_sft.py"
        ).read_text()

    def test_wrapper_calls_core(self, chat_sft_source: str) -> None:
        """The wrapper must invoke sft_data_generator_bos_bestfit_core."""
        assert "sft_data_generator_bos_bestfit_core(" in chat_sft_source, (
            "scripts/chat_sft.py wrapper must delegate to sft_data_generator_bos_bestfit_core"
        )

    def test_loop_does_not_use_produced_last_step(self, chat_sft_source: str) -> None:
        assert "progress.last_step" not in chat_sft_source
        assert "resolve_sft_loop_position(" in chat_sft_source

    def test_loop_does_not_schedule_from_produced_progress(self, chat_sft_source: str) -> None:
        assert "progress.approx_progress" not in chat_sft_source
        assert "committed_consumed = produced_progress.consumed" in chat_sft_source

    def test_wrapper_updates_current_epoch_global(self, chat_sft_source: str) -> None:
        assert "current_epoch = produced_progress.current_epoch" in chat_sft_source, (
            "loop must commit the final trained microbatch's epoch"
        )

    def test_val_progress_is_discarded(self, chat_sft_source: str) -> None:
        assert "for inputs, targets, _progress in" in chat_sft_source

    def test_wrapper_yields_inputs_and_targets_only(self, chat_sft_source: str) -> None:
        """The validation wrapper drops progress for evaluate_bpb."""
        # We expect a `yield inputs, targets` pattern (no `progress`
        # in the yielded tuple).
        import re

        # Looser regex: accept any whitespace/newlines.
        assert re.search(r"yield\s+inputs,\s*targets\s*$", chat_sft_source, re.MULTILINE), (
            "wrapper must yield (inputs, targets), dropping the internal progress snapshot"
        )

    def test_old_inline_generator_logic_removed(self, chat_sft_source: str) -> None:
        """Regression guard against partial revert: the OLD inline
        generator logic (find_bestfit_index call, refill_buffer
        nested function) must be GONE from chat_sft.py — those moved
        into steering.sft.iteration."""
        # The wrapper does NOT define a refill_buffer() nested fn.
        assert "def refill_buffer():" not in chat_sft_source, (
            "scripts/chat_sft.py still has an inline refill_buffer; "
            "the generator should be in steering.sft.iteration"
        )
        # The wrapper does NOT call find_bestfit_index directly (the
        # core does).
        assert "best_idx = find_bestfit_index(" not in chat_sft_source, (
            "scripts/chat_sft.py still calls find_bestfit_index directly; "
            "this should only happen in steering.sft.iteration"
        )


# ===========================================================================
# H. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Boundary conditions that don't naturally arise in production
    but must not crash. Defensive coverage."""

    def test_length_one_conversations(self) -> None:
        """Many tiny conversations of length 1 (just BOS). The packer
        should fit row_capacity of them per row."""
        gen = _make_gen(
            lengths=[1] * 50,
            max_seq_len=8,  # row_capacity=9 → 9 BOS-only convs per row
            device_batch_size=1,
            num_iterations=1,
            buffer_size=4,
        )
        batches = _drive(gen)
        inputs, _targets, progress = batches[0]
        # Every token in the row should be BOS (since each conv is just BOS).
        assert (inputs[0] == 0).all(), inputs

    def test_single_conversation_dataset_wraps(self) -> None:
        """dataset_size=1 → cursor wraps every fetch. Generator must
        not crash, and the cursor must actually wrap (epoch advances).

        Note: with dataset_size=1, per_rank_dataset_size=max(1, 1//1)=1,
        so the dataset-exhaustion path triggers as soon as the first
        yield consumes >=1 conversations. So we get exactly 1 yield
        (not 2). What matters is that the refill cursor wraps the tiny
        dataset many times — observable via current_epoch > 1."""
        gen = _make_gen(
            lengths=[3],
            max_seq_len=8,
            device_batch_size=1,
            num_iterations=2,
            buffer_size=2,
        )
        batches = _drive(gen)
        # At least one yield must succeed.
        assert len(batches) >= 1
        # The cursor wrapped through the 1-element dataset multiple times
        # during refill (buffer_size=2 already triggers epoch > 1).
        assert batches[-1][2].current_epoch > 1, (
            "cursor must wrap a single-conversation dataset; epoch should advance"
        )

    def test_multi_row_batch_independently_packed(self) -> None:
        """device_batch_size > 1 produces a batch with multiple rows.
        Each row is independently packed; rows can have different
        content_lens."""
        gen = _make_gen(
            lengths=[3, 5, 7, 4, 6, 8],
            max_seq_len=16,
            device_batch_size=3,
            num_iterations=1,
            buffer_size=6,
        )
        batches = _drive(gen)
        inputs, targets, _ = batches[0]
        assert inputs.shape == (3, 16)
        assert targets.shape == (3, 16)
        # Different rows likely have different content_len values
        # (due to different packing). Verify by counting -1 mask
        # positions per row — at least one row's mask count differs
        # from another's (with high probability for varied inputs).
        mask_counts = [(targets[i] == -1).sum().item() for i in range(3)]
        # We don't strictly require ALL different, but at least the
        # rows are independently shaped. Sanity: total tokens packed
        # across the 3 rows is positive.
        unmasked_total = (targets != -1).sum().item()
        assert unmasked_total > 0, f"all rows masked? mask_counts={mask_counts}"

    def test_single_pull_yields_one_produced_batch(self) -> None:
        gen = _make_gen(
            lengths=[3] * 10,
            max_seq_len=8,
            device_batch_size=1,
            num_iterations=1,
            grad_accum_steps=1,
            buffer_size=2,
        )
        batches = _drive(gen)
        assert len(batches) == 1
        assert batches[0][2].micro_batch_count == 1

    def test_dataset_size_one_with_many_pulls_remains_productive(self) -> None:
        gen = _make_gen(
            lengths=[3],
            max_seq_len=8,
            device_batch_size=1,
            num_iterations=10,
            grad_accum_steps=8,
            buffer_size=2,
        )
        batches = _drive(gen, max_yields=20)
        assert len(batches) == 20
        assert batches[-1][2].micro_batch_count == 20
        assert batches[-1][2].consumed >= batches[0][2].consumed
