"""Resume-from-step tests for the encoder-decoder SFT training loop.

Pretraining supports ``resume_from_step`` via
``conditional_encdec_train.py``.  SFT previously did not: its main loop
hard-coded ``step = 0`` and never read the optimizer state out of the
checkpoint.  This file pins the invariants for PR 3 (``SFT resume_from_step``):

1. **Cursor recovery** — ``make_sft_data_generator`` restores each rank's
   next logical cursor and ordered best-fit buffer, including variable-length
   conversations that are popped out of input order.
2. **Checkpoint loop-state** — SFT checkpoints carry scalar training state
   plus a JSON-safe per-rank dataloader-state envelope under ``loop_state``.
3. **Resume round-trip** — training two steps, saving, loading, then
   training two more steps reaches the same model state as training
   four steps continuously, within an extremely tight numerical
   tolerance.  If this drifts, something is wrong with how state is
   being captured or restored.
4. **Config surface** — ``resume_from_step`` defaults to ``-1`` (fresh
   start), rejects ``0`` (no such thing as resuming from step 0), and
   accepts any positive integer.

The first three classes use the shared helpers from
``tests.steering.test_encdec_sft_conditioned_integration`` so the data setup
stays consistent with PR 1's integration suite; the fourth reads the
shipped SFT config file directly.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from loguru import logger

import nanochat.flash_attention as _fa_mod
from errors import DataLoadError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _assert_sft_resume_artifact_coverage,
    _build_sft_checkpoint_dir,
    _restore_loop_state,
    _sft_step_batch_iterator,
    _validate_sft_dataloader_state,
)
from steering._common.resume import resolve_resume_step
from steering.conditional.encoder_decoder.dataloader import (
    build_dataloader_state_envelope,
    select_dataloader_rank_state,
)
from steering.profiling.markers import TimingRecorder
from steering.sft.loss_mask import LossMaskMode
from tests.encdec_test_helpers import setup_optimizer_simple
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _build_tiny_encdec_matching_sft,
    _make_gen,
    _run_one_training_step,
    build_synthetic_features,
    patch_load_sft_features,
)

# A realistic ``sft_artifact_coverage`` payload: nested fingerprints on BOTH
# sides, so the guard is exercised against the same shape production writes
# (and the same shape a JSON round-trip has to preserve).
_ARTIFACT_COVERAGE: dict[str, object] = {
    "dataset_size": 2000,
    "conversion_total_conversations": 2000,
    "parquet_mixture_fingerprint": {
        "mixture": "safety",
        "split": "train",
        "tasks": [{"class_name": "SafetyTask", "length": 2000}],
        "len_mixture": 2000,
        "index_map_sha256": "a" * 64,
    },
    "feature_mixture_fingerprint": {
        "mixture": "safety",
        "split": "train",
        "tasks": [{"class_name": "SafetyTask", "length": 2000}],
        "len_mixture": 2000,
        "index_map_sha256": "a" * 64,
    },
}


def _buffered_rank_state(*, consumed: int = 2) -> dict[str, object]:
    return {
        "ddp_rank": 0,
        "ddp_world_size": 1,
        "consumed": consumed,
        "next_cursor": consumed + 2,
        "buffered_cursor_positions": [consumed, consumed + 1],
        "buffer_size": 100,
        "device_batch_size": 1,
        "sequence_length": 64,
        "grad_accum_steps": 1,
        "min_loss_tokens": 1,
        "loss_mask_mode": "full_sequence",
    }


def _buffered_envelope(local_state: dict[str, object]) -> dict[str, object]:
    return {"ddp_world_size": 1, "rank_states": [local_state]}


def _gloo_sft_state_envelope_target(rank: int, world_size: int) -> dict[str, object]:
    local_state = {
        **_buffered_rank_state(consumed=(rank + 1) * world_size),
        "ddp_rank": rank,
        "ddp_world_size": world_size,
        "next_cursor": rank + 4 * world_size,
        "buffered_cursor_positions": [rank + 2 * world_size],
    }
    envelope = build_dataloader_state_envelope(
        local_state,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    selected = select_dataloader_rank_state(
        envelope,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    return {"envelope": envelope, "selected": selected}


def _assert_sft_batches_exact(actual: tuple, expected: tuple) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(expected_value, t.Tensor):
            assert isinstance(actual_value, t.Tensor)
            assert t.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value

# ---------------------------------------------------------------------------
# Attention backend override
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """Force SDPA attention for every test in this module.

    Same rationale as the fixture in
    ``tests/test_encdec_sft_conditioned_integration.py``:
    ``flex_attention`` does not support backward on CPU (the Triton
    backend raises an Inductor ``IndexError`` inside ``_compiled_flex``),
    so any test that runs ``model(...)`` ``.backward()`` on CPU must
    first route attention through SDPA.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Cursor recovery
# ---------------------------------------------------------------------------


class TestSFTFreshStateBoundary:
    """Fresh state starts at zero; scalar-only resume state is forbidden."""

    def test_fresh_consumed_starts_at_rank_zero(self) -> None:
        """A fresh ``SFTState()`` (consumed == 0) must yield examples
        in the default rank order starting at index 0."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=32, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        sft_state = SFTState()
        assert sft_state.consumed == 0

        with patch_load_sft_features(features):
            gen = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=sft_state,
            )
            _batch = next(gen)

        # After the first next(gen), at least one conversation has been
        # popped from the buffer.  ``consumed`` advances by at least
        # ``ddp_world_size`` (= 1 in single-rank tests) per pop.
        assert sft_state.consumed >= 1

    @pytest.mark.parametrize("consumed", [1, 12])
    def test_scalar_only_resume_fails_closed(self, consumed: int) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            generator = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=SFTState(consumed=consumed),
            )
            with pytest.raises(DataLoadError, match="cannot resume.*consumed alone"):
                next(generator)

    @pytest.mark.parametrize("consumed", [True, -1, 1.0, "1"])
    def test_invalid_fresh_consumed_domain_fails_closed(self, consumed: object) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            generator = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=SFTState(consumed=cast(Any, consumed)),
            )
            with pytest.raises(TypeError, match="NonNegativeInt"):
                next(generator)

    @pytest.mark.parametrize(("ddp_rank", "ddp_world_size"), [(-1, 1), (1, 1), (0, 0)])
    def test_ddp_rank_and_world_size_relation(
        self, ddp_rank: int, ddp_world_size: int
    ) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        with pytest.raises(DataLoadError, match="0 <= ddp_rank < ddp_world_size"):
            _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
            )


class _VariableLengthTask(SingleTurnTask):
    def __init__(self, rendered_lengths: tuple[int, ...]) -> None:
        super().__init__(n=len(rendered_lengths))
        self._rendered_lengths = rendered_lengths

    def get_example(self, index: int) -> dict:
        # Mock rendering contributes BOS + two role markers + one user token.
        assistant_chars = self._rendered_lengths[index] - 4
        return {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a" * assistant_chars},
            ]
        }


def _assert_batches_equal(left: tuple, right: tuple) -> None:
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right, strict=True):
        if left_item is None or right_item is None:
            assert left_item is right_item
        else:
            assert t.equal(left_item, right_item)


class TestSFTBufferedCursorRecovery:
    def test_variable_length_best_fit_resume_is_tensor_exact(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = _VariableLengthTask((10, 25, 15, 12))
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            continuous_state = SFTState()
            continuous = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=24,
                buffer_size=3,
                sft_state=continuous_state,
            )
            next(continuous)
            assert continuous_state.dataloader_state == {
                "ddp_rank": 0,
                "ddp_world_size": 1,
                "consumed": 1,
                "next_cursor": 3,
                "buffered_cursor_positions": [0, 2],
                "buffer_size": 3,
                "device_batch_size": 1,
                "sequence_length": 24,
                "grad_accum_steps": 1,
                "min_loss_tokens": 1,
                "loss_mask_mode": "full_sequence",
            }
            checkpoint_state = json.loads(json.dumps(continuous_state.dataloader_state))
            expected_next = next(continuous)

        with patch_load_sft_features(features):
            resumed_state = SFTState(
                consumed=checkpoint_state["consumed"],
                dataloader_state=checkpoint_state,
            )
            resumed = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=24,
                buffer_size=3,
                sft_state=resumed_state,
            )
            actual_next = next(resumed)

        _assert_batches_equal(expected_next, actual_next)
        assert resumed_state.dataloader_state == continuous_state.dataloader_state

    def test_filtered_cursor_gap_is_preserved_on_resume(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = _VariableLengthTask((5, 25, 15, 12))
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            continuous_state = SFTState()
            continuous = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=24,
                buffer_size=3,
                min_loss_tokens=6,
                sft_state=continuous_state,
            )
            next(continuous)
            assert continuous_state.dataloader_state is not None
            assert continuous_state.dataloader_state["next_cursor"] == 4
            assert continuous_state.dataloader_state["buffered_cursor_positions"] == [2, 3]
            checkpoint_state = copy.deepcopy(continuous_state.dataloader_state)
            expected_next = next(continuous)

        with patch_load_sft_features(features):
            resumed_state = SFTState(
                consumed=checkpoint_state["consumed"],
                dataloader_state=checkpoint_state,
            )
            resumed = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=24,
                buffer_size=3,
                min_loss_tokens=6,
                sft_state=resumed_state,
            )
            actual_next = next(resumed)

        _assert_batches_equal(expected_next, actual_next)
        assert resumed_state.dataloader_state == continuous_state.dataloader_state

    def test_asymmetric_ddp_ranks_restore_their_own_buffer(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = _VariableLengthTask((10, 25, 10, 25, 10, 25))
        features = build_synthetic_features(task, tokenizer)
        saved_states: list[dict[str, object]] = []
        expected_batches: list[tuple] = []

        for rank in range(2):
            with patch_load_sft_features(features):
                state = SFTState()
                generator = _make_gen(
                    task,
                    tokenizer,
                    alignment_probe_samples=0,
                    B=1,
                    T=24,
                    buffer_size=3,
                    ddp_rank=rank,
                    ddp_world_size=2,
                    sft_state=state,
                )
                next(generator)
                assert state.dataloader_state is not None
                saved_states.append(copy.deepcopy(state.dataloader_state))
                expected_batches.append(next(generator))

        assert saved_states[0]["consumed"] == 4
        assert saved_states[1]["consumed"] == 2
        envelope = {"ddp_world_size": 2, "rank_states": saved_states}

        for rank in range(2):
            local_state = select_dataloader_rank_state(
                envelope,
                ddp_rank=rank,
                ddp_world_size=2,
            )
            assert local_state is not None
            with patch_load_sft_features(features):
                resumed_state = SFTState(
                    consumed=local_state["consumed"],
                    dataloader_state=local_state,
                )
                resumed = _make_gen(
                    task,
                    tokenizer,
                    alignment_probe_samples=0,
                    B=1,
                    T=24,
                    buffer_size=3,
                    ddp_rank=rank,
                    ddp_world_size=2,
                    sft_state=resumed_state,
                )
                actual_batch = next(resumed)
            _assert_batches_equal(expected_batches[rank], actual_batch)

    def test_real_gloo_checkpoint_gathers_every_rank_state(
        self, gloo_multiprocess
    ) -> None:
        results = gloo_multiprocess(2, _gloo_sft_state_envelope_target)
        expected_consumed = [2, 4]
        for rank, result in enumerate(results):
            envelope = result["envelope"]
            assert [state["consumed"] for state in envelope["rank_states"]] == (
                expected_consumed
            )
            assert result["selected"] == envelope["rank_states"][rank]

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"buffered_cursor_positions": (2, 3)}, "must be a list"),
            ({"buffered_cursor_positions": [2, 1]}, "strictly increasing"),
            ({"buffered_cursor_positions": [0, 4]}, "precede next_cursor"),
            ({"ddp_rank": 1}, "ddp_rank"),
            ({"ddp_world_size": 2}, "ddp_world_size"),
            ({"buffer_size": 99}, "buffer_size"),
            ({"device_batch_size": 2}, "device_batch_size"),
            ({"sequence_length": 63}, "sequence_length"),
            ({"grad_accum_steps": 2}, "grad_accum_steps"),
            ({"min_loss_tokens": 2}, "min_loss_tokens"),
            ({"loss_mask_mode": "completion_only"}, "loss_mask_mode"),
            ({"consumed": 4}, "more consumed and buffered"),
        ],
    )
    def test_malformed_or_incompatible_state_fails_closed(
        self, mutation: dict[str, object], message: str
    ) -> None:
        state = _buffered_rank_state()
        state.update(mutation)
        with pytest.raises(DataLoadError, match=message):
            _validate_sft_dataloader_state(
                state,
                ddp_rank=0,
                ddp_world_size=1,
                buffer_size=100,
                device_batch_size=1,
                sequence_length=64,
                grad_accum_steps=1,
                min_loss_tokens=1,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            )

    @pytest.mark.parametrize(
        "mutation",
        [
            {"next_cursor": True},
            {"consumed": True},
            {"consumed": -1},
            {"buffered_cursor_positions": [2, True]},
        ],
    )
    def test_cursor_integers_use_nonnegative_int_phantom(
        self, mutation: dict[str, object]
    ) -> None:
        state = _buffered_rank_state()
        state.update(mutation)
        with pytest.raises(TypeError, match="NonNegativeInt"):
            _validate_sft_dataloader_state(
                state,
                ddp_rank=0,
                ddp_world_size=1,
                buffer_size=100,
                device_batch_size=1,
                sequence_length=64,
                grad_accum_steps=1,
                min_loss_tokens=1,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            )

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"consumed": 3}, "divisible"),
            ({"next_cursor": 4}, "next_cursor=4.*rank 1"),
            ({"buffered_cursor_positions": [2]}, "position 2.*rank 1"),
        ],
    )
    def test_two_rank_cursor_congruence_fails_closed(
        self, mutation: dict[str, object], message: str
    ) -> None:
        state = {
            **_buffered_rank_state(),
            "ddp_rank": 1,
            "ddp_world_size": 2,
            "consumed": 2,
            "next_cursor": 5,
            "buffered_cursor_positions": [3],
        }
        state.update(mutation)
        with pytest.raises(DataLoadError, match=message):
            _validate_sft_dataloader_state(
                state,
                ddp_rank=1,
                ddp_world_size=2,
                buffer_size=100,
                device_batch_size=1,
                sequence_length=64,
                grad_accum_steps=1,
                min_loss_tokens=1,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            )

    def test_buffer_larger_than_config_fails_closed(self) -> None:
        state = {
            **_buffered_rank_state(),
            "next_cursor": 101,
            "buffered_cursor_positions": list(range(101)),
        }
        with pytest.raises(DataLoadError, match="more buffered positions"):
            _validate_sft_dataloader_state(
                state,
                ddp_rank=0,
                ddp_world_size=1,
                buffer_size=100,
                device_batch_size=1,
                sequence_length=64,
                grad_accum_steps=1,
                min_loss_tokens=1,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            )

    def test_state_object_and_nested_consumed_must_match(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        local_state = _buffered_rank_state(consumed=2)
        with patch_load_sft_features(features):
            generator = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=SFTState(consumed=1, dataloader_state=local_state),
            )
            with pytest.raises(DataLoadError, match="consumed disagrees"):
                next(generator)

    def test_replayed_position_that_becomes_filtered_fails_closed(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = _VariableLengthTask((5, 10, 10))
        features = build_synthetic_features(task, tokenizer)
        local_state = {
            **_buffered_rank_state(consumed=0),
            "next_cursor": 1,
            "buffered_cursor_positions": [0],
            "buffer_size": 3,
            "sequence_length": 24,
            "min_loss_tokens": 6,
        }
        with patch_load_sft_features(features):
            generator = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=24,
                buffer_size=3,
                min_loss_tokens=6,
                sft_state=SFTState(consumed=0, dataloader_state=local_state),
            )
            with pytest.raises(DataLoadError, match="buffered conversation no longer"):
                next(generator)


class TestSFTBatchPrefetchResumeParity:
    """Prefetch must not change batches or advance checkpoint-visible state."""

    def test_prefetch_matches_synchronous_batches_and_state_exactly(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=32, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        synchronous_state = SFTState()
        producer_state = SFTState()
        consumer_state = SFTState()
        with patch_load_sft_features(features):
            synchronous = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=synchronous_state,
            )
            raw_prefetched = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=producer_state,
                profile_markers=True,
            )
            recorder = TimingRecorder()
            with _sft_step_batch_iterator(
                raw_prefetched,
                producer_state=producer_state,
                consumer_state=consumer_state,
                batch_count=4,
                prefetch_batches=2,
                cuda_device=None,
                timing_recorder=recorder,
                timing_rank=2,
                timing_step=11,
            ) as prefetched:
                for _ in range(4):
                    _assert_sft_batches_exact(next(prefetched), next(synchronous))
                    assert consumer_state == synchronous_state

            assert recorder.rows()
            for row in recorder.rows():
                assert row["rank"] == 2
                assert row["step"] == 11
                assert row["micro_step"] in range(4)
            assert [
                row["micro_step"]
                for row in recorder.rows()
                if row["name"] == "sft/build_text_tensors"
            ] == [0, 1, 2, 3]

    def test_discarded_lookahead_regenerates_exact_next_batch_on_resume(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=32, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            baseline_state = SFTState()
            baseline = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=baseline_state,
            )
            next(baseline)
            next(baseline)
            expected_third = next(baseline)
            expected_consumed = baseline_state.consumed

            producer_state = SFTState()
            committed_state = SFTState()
            raw = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=producer_state,
            )
            with _sft_step_batch_iterator(
                raw,
                producer_state=producer_state,
                consumer_state=committed_state,
                batch_count=4,
                prefetch_batches=2,
                cuda_device=None,
            ) as prefetched:
                next(prefetched)
                next(prefetched)
                checkpoint_consumed = committed_state.consumed
                checkpoint_dataloader_state = copy.deepcopy(
                    committed_state.dataloader_state
                )
                assert checkpoint_dataloader_state is not None

            resumed_producer = SFTState(
                consumed=checkpoint_consumed,
                dataloader_state=copy.deepcopy(checkpoint_dataloader_state),
            )
            resumed_consumer = SFTState(
                consumed=checkpoint_consumed,
                dataloader_state=copy.deepcopy(checkpoint_dataloader_state),
            )
            resumed_raw = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=resumed_producer,
            )
            with _sft_step_batch_iterator(
                resumed_raw,
                producer_state=resumed_producer,
                consumer_state=resumed_consumer,
                batch_count=1,
                prefetch_batches=2,
                cuda_device=None,
            ) as resumed:
                actual_third = next(resumed)

        _assert_sft_batches_exact(actual_third, expected_third)
        assert resumed_consumer.consumed == expected_consumed
        assert resumed_consumer.dataloader_state == baseline_state.dataloader_state


# ---------------------------------------------------------------------------
# Checkpoint loop-state round-trip
# ---------------------------------------------------------------------------


class TestSFTCheckpointLoopState:
    """SFT checkpoint meta must carry ``loop_state`` with the scalar
    training state needed for a resume.

    These tests hit ``nanochat.checkpoint_manager.save_checkpoint`` and
    ``load_checkpoint`` directly — no SFT main loop — to pin the
    on-disk format independently of whichever script produced it.
    """

    def test_save_load_roundtrip_preserves_loop_state(self, tmp_path: Path) -> None:
        """Every field we write into ``loop_state`` must survive the
        JSON round-trip that ``save_checkpoint`` performs."""
        tiny_model_state: dict[str, t.Tensor] = {
            "dummy": t.zeros(1, dtype=t.float32)
        }
        meta = {
            "step": 42,
            "dataset": "safety",
            "user_config": {"lr": 0.001},
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 1.2345,
                "smooth_fp_loss": 0.0,
                "min_val_bpb": 0.9876,
                "total_training_time": 123.456,
                "sft_consumed": 42 * 8,  # step * some batch-worth
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=42 * 8)
                ),
                "sft_artifact_coverage": _ARTIFACT_COVERAGE,
            },
        }
        save_checkpoint(
            str(tmp_path / "ckpt"),
            step=42,
            model_data=tiny_model_state,
            optimizer_data=None,
            meta_data=meta,
        )

        _model_data, _opt_data, loaded_meta = load_checkpoint(
            str(tmp_path / "ckpt"), step=42, device="cpu"
        )

        assert loaded_meta["step"] == 42
        assert loaded_meta["dataset"] == "safety"
        loop = loaded_meta["loop_state"]
        assert loop["smooth_train_loss"] == pytest.approx(1.2345)
        assert loop["min_val_bpb"] == pytest.approx(0.9876)
        assert loop["total_training_time"] == pytest.approx(123.456)
        assert loop["sft_consumed"] == 42 * 8
        assert loop["sft_dataloader_state"] == _buffered_envelope(
            _buffered_rank_state(consumed=42 * 8)
        )
        assert loop["sft_artifact_coverage"] == _ARTIFACT_COVERAGE

    def test_save_load_roundtrip_preserves_optimizer_state(
        self, tmp_path: Path
    ) -> None:
        """When the checkpoint is written with a real optimizer state,
        ``load_checkpoint(..., load_optimizer=True)`` must return a dict
        whose contents can be loaded back into a fresh optimizer via
        ``optimizer.load_state_dict(...)``.

        This is the invariant SFT resume relies on: after a resume the
        Adam (and Muon) moments must survive across save/load so the
        next training step does not see a cold optimizer.
        """
        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model, lr=0.01)

        # Take one training step so the optimizer has non-trivial state.
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)
        _run_one_training_step(model, batch, optimizer)

        # Capture post-step optimizer state deep copy.
        state_before = copy.deepcopy(optimizer.state_dict())

        save_checkpoint(
            str(tmp_path / "ckpt"),
            step=1,
            model_data=model.state_dict(),
            optimizer_data=optimizer.state_dict(),
            meta_data={
                "step": 1,
                "loop_state": {
                    "sft_cursor_semantics": "buffered_v2",
                    "smooth_train_loss": 1.0,
                    "smooth_fp_loss": 0.0,
                    "min_val_bpb": float("inf"),
                    "total_training_time": 0.1,
                    "sft_consumed": 2,
                    "sft_dataloader_state": _buffered_envelope(
                        _buffered_rank_state(consumed=2)
                    ),
                    "sft_artifact_coverage": _ARTIFACT_COVERAGE,
                },
            },
            rank=0,
        )

        _model_data, optimizer_data, _meta = load_checkpoint(
            str(tmp_path / "ckpt"), step=1, device="cpu", load_optimizer=True
        )
        assert optimizer_data is not None

        # Load the round-tripped optimizer state into a fresh optimizer
        # and assert the state survived.
        fresh_optimizer = setup_optimizer_simple(model, lr=0.01)
        fresh_optimizer.load_state_dict(optimizer_data)

        def _compare(a, b) -> bool:
            if isinstance(a, t.Tensor) and isinstance(b, t.Tensor):
                return t.equal(a, b)
            return a == b

        for group_before, group_after in zip(
            state_before["param_groups"],
            fresh_optimizer.state_dict()["param_groups"],
            strict=True,
        ):
            # Group hyperparameters must round-trip.
            assert group_before["lr"] == pytest.approx(group_after["lr"])

        # Per-parameter state (moments) must round-trip bitwise.
        before_states = state_before["state"]
        after_states = fresh_optimizer.state_dict()["state"]
        assert set(before_states.keys()) == set(after_states.keys())
        for key in before_states:
            before_inner = before_states[key]
            after_inner = after_states[key]
            assert set(before_inner.keys()) == set(after_inner.keys())
            for inner_key in before_inner:
                assert _compare(before_inner[inner_key], after_inner[inner_key]), (
                    f"Optimizer state mismatch at [{key}][{inner_key}]"
                )


# ---------------------------------------------------------------------------
# Resume round-trip: interrupted run === continuous run
# ---------------------------------------------------------------------------


class TestSFTResumeRoundtrip:
    """A save → load → continue sequence must produce the same training
    trajectory as an uninterrupted run with the same seed.

    This is the single strongest end-to-end guarantee of PR 3: if
    anything drifts (optimizer state, RNG, data cursor, EMA of loss,
    learning-rate schedule) the final model state or loss sequence will
    not match.  Because we don't want the test to depend on Muon's
    internal RNG, the tiny model uses plain AdamW via
    ``setup_optimizer_simple`` from the encdec test helpers.
    """

    @staticmethod
    def _manual_snapshot(model) -> dict[str, t.Tensor]:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def _tensors_equal(
        snap_a: dict[str, t.Tensor],
        snap_b: dict[str, t.Tensor],
        atol: float = 1e-6,
    ) -> tuple[bool, str]:
        if snap_a.keys() != snap_b.keys():
            return False, f"keys differ: {snap_a.keys() ^ snap_b.keys()}"
        for name, tensor_a in snap_a.items():
            tensor_b = snap_b[name]
            if tensor_a.shape != tensor_b.shape:
                return False, f"shape mismatch at {name}"
            if not t.allclose(tensor_a, tensor_b, atol=atol):
                diff = (tensor_a - tensor_b).abs().max().item()
                return False, f"value mismatch at {name}: max|diff|={diff}"
        return True, ""

    def _continuous_run(
        self, tmp_path: Path, *, n_steps: int, seed: int
    ) -> tuple[dict[str, t.Tensor], int, list[float]]:
        """Run ``n_steps`` of training with no interruption and return
        (final_state_dict, final_consumed, losses)."""
        t.manual_seed(seed)
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=32, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model, lr=0.01)
        sft_state = SFTState()

        losses: list[float] = []
        with patch_load_sft_features(features):
            gen = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=sft_state,
            )
            for _ in range(n_steps):
                batch = next(gen)
                losses.append(_run_one_training_step(model, batch, optimizer))
        return self._manual_snapshot(model), sft_state.consumed, losses

    def _interrupted_run(
        self,
        tmp_path: Path,
        *,
        first_half_steps: int,
        second_half_steps: int,
        seed: int,
    ) -> tuple[dict[str, t.Tensor], int, list[float]]:
        """Run ``first_half_steps`` training steps, save model+optimizer+
        loop state, reload fresh, run ``second_half_steps`` more, return
        the combined (final_state_dict, final_consumed, losses)."""
        t.manual_seed(seed)
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=32, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model, lr=0.01)
        sft_state = SFTState()

        losses_a: list[float] = []
        with patch_load_sft_features(features):
            gen = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=sft_state,
            )
            for _ in range(first_half_steps):
                batch = next(gen)
                losses_a.append(_run_one_training_step(model, batch, optimizer))

        # Save.
        save_dir = tmp_path / "sft_ckpt"
        assert sft_state.dataloader_state is not None
        save_checkpoint(
            str(save_dir),
            step=first_half_steps,
            model_data=model.state_dict(),
            optimizer_data=optimizer.state_dict(),
            meta_data={
                "step": first_half_steps,
                "loop_state": {
                    "sft_cursor_semantics": "buffered_v2",
                    "smooth_train_loss": 0.0,
                    "smooth_fp_loss": 0.0,
                    "min_val_bpb": float("inf"),
                    "total_training_time": 0.0,
                    "sft_consumed": sft_state.consumed,
                    "sft_dataloader_state": _buffered_envelope(
                        copy.deepcopy(sft_state.dataloader_state)
                    ),
                    "sft_artifact_coverage": _ARTIFACT_COVERAGE,
                },
            },
            rank=0,
        )

        saved_consumed = sft_state.consumed

        # Load: fresh model + optimizer, load state back in.
        del model, optimizer, sft_state, gen

        model_b = _build_tiny_encdec_matching_sft()
        optimizer_b = setup_optimizer_simple(model_b, lr=0.01)

        model_data, optimizer_data, meta_data = load_checkpoint(
            str(save_dir), step=first_half_steps, device="cpu", load_optimizer=True
        )
        model_b.load_state_dict(model_data)
        assert optimizer_data is not None
        optimizer_b.load_state_dict(optimizer_data)
        local_state = meta_data["loop_state"]["sft_dataloader_state"]["rank_states"][0]
        sft_state_b = SFTState(
            consumed=local_state["consumed"],
            dataloader_state=local_state,
        )
        assert sft_state_b.consumed == saved_consumed

        losses_b: list[float] = []
        with patch_load_sft_features(features):
            gen_b = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=64,
                sft_state=sft_state_b,
            )
            for _ in range(second_half_steps):
                batch = next(gen_b)
                losses_b.append(_run_one_training_step(model_b, batch, optimizer_b))

        return (
            self._manual_snapshot(model_b),
            sft_state_b.consumed,
            losses_a + losses_b,
        )

    def test_resumed_run_matches_continuous_run_model_state(
        self, tmp_path: Path
    ) -> None:
        """Bitwise (up to 1e-6) model-parameter equivalence between
        a continuous 4-step run and a 2+2 split with save/load in the
        middle."""
        continuous_state, continuous_consumed, _ = self._continuous_run(
            tmp_path, n_steps=4, seed=1234
        )
        interrupted_state, interrupted_consumed, _ = self._interrupted_run(
            tmp_path, first_half_steps=2, second_half_steps=2, seed=1234
        )

        assert continuous_consumed == interrupted_consumed
        equal, reason = self._tensors_equal(continuous_state, interrupted_state)
        assert equal, (
            f"Model state must match after save/load/continue: {reason}"
        )

    def test_resumed_run_matches_continuous_loss_trajectory(
        self, tmp_path: Path
    ) -> None:
        """The per-step loss values must match element-for-element
        between a continuous run and an interrupted one.  If they
        diverge, something is wrong with how post-save state is being
        restored (or the data cursor drifted)."""
        _, _, continuous_losses = self._continuous_run(
            tmp_path, n_steps=4, seed=2024
        )
        _, _, interrupted_losses = self._interrupted_run(
            tmp_path, first_half_steps=2, second_half_steps=2, seed=2024
        )

        assert len(continuous_losses) == len(interrupted_losses) == 4
        for step_idx, (a, b) in enumerate(
            zip(continuous_losses, interrupted_losses, strict=True)
        ):
            assert a == pytest.approx(b, rel=1e-5, abs=1e-6), (
                f"Loss diverged at step {step_idx}: "
                f"continuous={a}, interrupted={b}"
            )


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestSFTResumeConfig:
    """``configs/training/conditional_encdec_sft.yaml`` must carry a
    ``resume_from_step`` default so operators can opt into resume via
    a CLI override.  Matching pretraining: ``-1`` means fresh, positive
    integers trigger the resume path."""

    @pytest.fixture()
    def cfg(self, hydra_compose):
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        return hydra_compose(config_dir, "conditional_encdec_sft")

    def test_resume_from_step_default_is_negative_one(self, cfg) -> None:
        assert cfg.resume_from_step == -1, (
            f"resume_from_step default must be -1 (fresh start), "
            f"got {cfg.resume_from_step}"
        )


# ---------------------------------------------------------------------------
# Resume helper functions (pure logic extracted from main())
# ---------------------------------------------------------------------------


class TestBuildSFTCheckpointDir:
    """``_build_sft_checkpoint_dir`` composes the per-dataset leaf
    directory name under the encdec pretrain checkpoint dir."""

    def test_conditioned_suffix(self) -> None:
        path = _build_sft_checkpoint_dir("/base/d26/s128/t2048.9", "safety")
        assert path.endswith("safety_sft_conditioned_checkpoints")
        assert path.startswith("/base/d26/s128/t2048.9")

    def test_dataset_name_in_path(self) -> None:
        for dataset in ("safety", "capabilities", "safety_pku", "code", "math"):
            path = _build_sft_checkpoint_dir("/base", dataset)
            assert path.endswith(f"{dataset}_sft_conditioned_checkpoints"), (
                f"dataset {dataset!r} did not appear in {path}"
            )


class TestResolveResumeStep:
    """``resolve_resume_step`` validates ``cfg.resume_from_step``.

    The function returns ``None`` for a fresh start (``-1``), an int
    for a valid resume, and raises ``AssertionError`` for anything
    else (``0`` and all negative values except ``-1``).
    """

    def test_minus_one_is_fresh_start(self) -> None:
        assert resolve_resume_step(-1) is None

    def test_positive_returns_step_number(self) -> None:
        for step in (1, 100, 5000, 1_000_000):
            assert resolve_resume_step(step) == step

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="resume_from_step"):
            resolve_resume_step(0)

    def test_negative_other_than_minus_one_is_rejected(self) -> None:
        for bad in (-2, -10, -1000):
            with pytest.raises(AssertionError, match="resume_from_step"):
                resolve_resume_step(bad)


class TestRestoreLoopState:
    """``_restore_loop_state`` recovers the training loop scalars from
    a loaded SFT checkpoint meta (or returns fresh defaults)."""

    def test_none_returns_fresh_defaults(self) -> None:
        step, smooth, smooth_fp, min_val, total_time, dataloader_state = (
            _restore_loop_state(None)
        )
        assert step == 0
        assert smooth == 0.0
        assert smooth_fp == 0.0
        assert min_val == float("inf")
        assert total_time == 0.0
        assert dataloader_state is None

    def test_full_meta_is_restored(self) -> None:
        expected_state = _buffered_envelope(_buffered_rank_state(consumed=12000))
        meta = {
            "step": 1500,
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 1.234,
                "smooth_fp_loss": 0.456,
                "min_val_bpb": 0.987,
                "total_training_time": 321.0,
                "sft_consumed": 12000,
                "sft_dataloader_state": expected_state,
                "sft_artifact_coverage": _ARTIFACT_COVERAGE,
            },
        }
        step, smooth, smooth_fp, min_val, total_time, dataloader_state = (
            _restore_loop_state(meta)
        )
        assert step == 1500
        assert smooth == pytest.approx(1.234)
        assert smooth_fp == pytest.approx(0.456)
        assert min_val == pytest.approx(0.987)
        assert total_time == pytest.approx(321.0)
        assert dataloader_state == expected_state

    def test_historical_prefetched_cursor_schema_fails_closed(self) -> None:
        meta = {
            "step": 1500,
            "loop_state": {
                "smooth_train_loss": 1.234,
                "smooth_fp_loss": 0.456,
                "min_val_bpb": 0.987,
                "total_training_time": 321.0,
                "sft_consumed": 12000,
            },
        }
        with pytest.raises(
            DataLoadError,
            match="ordered best-fit buffer.*every DDP rank",
        ):
            _restore_loop_state(meta)

    def test_committed_v1_schema_fails_closed(self) -> None:
        meta = {
            "step": 100,
            "loop_state": {
                "sft_cursor_semantics": "committed_v1",
                "smooth_train_loss": 2.0,
                "smooth_fp_loss": 0.0,
                "min_val_bpb": 1.5,
                "total_training_time": 60.0,
                "sft_consumed": 200,
            },
        }
        with pytest.raises(DataLoadError, match="committed_v1.*cannot be reconstructed"):
            _restore_loop_state(meta)

    def test_missing_smooth_fp_loss_raises_keyerror(self) -> None:
        """Loop-state resume requires the current schema."""
        meta = {
            "step": 100,
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 2.0,
                "min_val_bpb": 1.5,
                "total_training_time": 60.0,
                "sft_consumed": 200,
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=200)
                ),
                "sft_artifact_coverage": _ARTIFACT_COVERAGE,
            },
        }
        with pytest.raises(KeyError, match="smooth_fp_loss"):
            _restore_loop_state(meta)

    @pytest.mark.parametrize("coverage", [None, [], "unpinned"])
    def test_buffered_v2_requires_mapping_artifact_coverage(
        self, coverage: object
    ) -> None:
        meta = {
            "step": 100,
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 2.0,
                "smooth_fp_loss": 0.0,
                "min_val_bpb": 1.5,
                "total_training_time": 60.0,
                "sft_consumed": 200,
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=200)
                ),
                "sft_artifact_coverage": coverage,
            },
        }
        with pytest.raises(DataLoadError, match="artifact_coverage must be a mapping"):
            _restore_loop_state(meta)

    def test_buffered_v2_missing_artifact_coverage_fails_closed(self) -> None:
        meta = {
            "step": 100,
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 2.0,
                "smooth_fp_loss": 0.0,
                "min_val_bpb": 1.5,
                "total_training_time": 60.0,
                "sft_consumed": 200,
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=200)
                ),
            },
        }
        with pytest.raises(KeyError, match="sft_artifact_coverage"):
            _restore_loop_state(meta)

    def test_step_is_int_coerced(self) -> None:
        """JSON round-trip can yield floats where ints are expected; the
        helper must coerce back to ``int`` so code downstream doesn't
        accidentally index with a float."""
        meta = {
            "step": 1500.0,  # stored as float in JSON
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 0.0,
                "smooth_fp_loss": 0.0,
                "min_val_bpb": 0.0,
                "total_training_time": 0.0,
                "sft_consumed": 12000.0,
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=12000)
                ),
                "sft_artifact_coverage": _ARTIFACT_COVERAGE,
            },
        }
        step, _, _, _, _, dataloader_state = _restore_loop_state(meta)
        assert isinstance(step, int)
        assert step == 1500
        assert dataloader_state is not None

    def test_missing_loop_state_key_raises_keyerror(self) -> None:
        """If someone loads an old (pre-PR-3) SFT checkpoint, the
        ``loop_state`` key will be missing and we must fail loud rather
        than silently reset state."""
        meta = {
            "step": 100,
            "dataset": "safety",
            "user_config": {},
            "min_val_bpb": 1.0,
            # NO loop_state key
        }
        with pytest.raises(KeyError, match="loop_state"):
            _restore_loop_state(meta)

    def test_missing_loop_state_sub_key_raises_keyerror(self) -> None:
        """If the loop_state is present but missing a required sub-key
        (schema mismatch from an even older version), fail loud."""
        meta = {
            "step": 100,
            "loop_state": {
                "sft_cursor_semantics": "buffered_v2",
                "smooth_train_loss": 0.0,
                # missing min_val_bpb, total_training_time, sft_consumed
                "sft_dataloader_state": _buffered_envelope(
                    _buffered_rank_state(consumed=100)
                ),
                "sft_artifact_coverage": _ARTIFACT_COVERAGE,
            },
        }
        with pytest.raises(KeyError):
            _restore_loop_state(meta)


class TestSFTResumeArtifactCoverage:
    """A resumed cursor must remain tied to its original artifact family."""

    @staticmethod
    def _saved_via_checkpoint(tmp_path: Path, coverage: dict[str, object]) -> dict:
        """Return the meta as ``load_checkpoint`` hands it back.

        ``load_checkpoint`` reads JSON, so the saved coverage is ALWAYS a
        distinct object from the live one. Passing the same object as both
        sides (the previous revision) made the accept test survive mutating
        the comparison to ``saved is not current`` — while production, which
        never has object identity, would have rejected every legitimate
        resume (review finding 11).
        """
        save_checkpoint(
            str(tmp_path / "ckpt"),
            step=7,
            model_data={"dummy": t.zeros(1, dtype=t.float32)},
            optimizer_data=None,
            meta_data={
                "step": 7,
                "loop_state": {
                    "sft_cursor_semantics": "buffered_v2",
                    "smooth_train_loss": 1.0,
                    "smooth_fp_loss": 0.0,
                    "min_val_bpb": 1.0,
                    "total_training_time": 1.0,
                    "sft_consumed": 8,
                    "sft_dataloader_state": _buffered_envelope(
                        _buffered_rank_state(consumed=8)
                    ),
                    "sft_artifact_coverage": coverage,
                },
            },
        )
        _model, _opt, meta = load_checkpoint(str(tmp_path / "ckpt"), step=7, device="cpu")
        return meta

    def test_roundtripped_matching_coverage_is_accepted(self, tmp_path: Path) -> None:
        meta = self._saved_via_checkpoint(tmp_path, _ARTIFACT_COVERAGE)
        saved = meta["loop_state"]["sft_artifact_coverage"]
        # The value really did survive a JSON round-trip as a NEW object.
        assert saved is not _ARTIFACT_COVERAGE
        assert saved == _ARTIFACT_COVERAGE

        _assert_sft_resume_artifact_coverage(meta, copy.deepcopy(_ARTIFACT_COVERAGE))

    def test_absent_coverage_warns_and_resumes(self, tmp_path: Path) -> None:
        """Every pre-#1311 checkpoint lacks the key; bricking them is worse.

        A multi-day 8-GPU run checkpointing before the key existed could
        otherwise only be "resumed" by training from scratch (finding 3).
        """
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            _assert_sft_resume_artifact_coverage(
                {"loop_state": {"sft_consumed": 8}}, _ARTIFACT_COVERAGE
            )
        finally:
            logger.remove(sink_id)

        assert any("no sft_artifact_coverage" in message for message in messages)

    def test_malformed_coverage_is_rejected(self) -> None:
        with pytest.raises(DataLoadError, match=r"malformed sft_artifact_coverage"):
            _assert_sft_resume_artifact_coverage(
                {"loop_state": {"sft_artifact_coverage": "2000"}}, _ARTIFACT_COVERAGE
            )

    def test_changed_dataset_size_is_rejected(self, tmp_path: Path) -> None:
        meta = self._saved_via_checkpoint(tmp_path, _ARTIFACT_COVERAGE)
        changed = dict(_ARTIFACT_COVERAGE, dataset_size=1_516_934)
        with pytest.raises(DataLoadError, match=r"artifact coverage mismatch"):
            _assert_sft_resume_artifact_coverage(meta, changed)

    def test_same_mixture_different_feature_extraction_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Finding 7: same parquet mixture, different feature side.

        Re-pointing ``feature_dir`` at another extraction of the SAME mixture
        (other layer / SAE width / pooling) leaves every parquet-side field
        identical, so only feature-side identity can catch it.
        """
        meta = self._saved_via_checkpoint(tmp_path, _ARTIFACT_COVERAGE)
        other_features = copy.deepcopy(_ARTIFACT_COVERAGE)
        fingerprint: Any = other_features["feature_mixture_fingerprint"]
        assert isinstance(fingerprint, dict)
        fingerprint["index_map_sha256"] = "b" * 64

        with pytest.raises(DataLoadError, match=r"artifact coverage mismatch"):
            _assert_sft_resume_artifact_coverage(meta, other_features)

    def test_absent_current_coverage_is_rejected(self, tmp_path: Path) -> None:
        meta = self._saved_via_checkpoint(tmp_path, _ARTIFACT_COVERAGE)
        with pytest.raises(DataLoadError, match=r"artifact coverage mismatch"):
            _assert_sft_resume_artifact_coverage(meta, None)


# ---------------------------------------------------------------------------
# Main-loop source inspection
# ---------------------------------------------------------------------------


class TestSFTMainResumeWiring:
    """Lightweight source inspection of ``scripts.train.conditional_encdec_sft``'s
    ``main()`` function to verify the resume logic stays wired up.

    These are not exact-string assertions like the brittle tests in
    ``tests/test_gradient_clipping.py`` — they check for the presence of
    *names* (``resume_from_step``, ``loop_state``) that must appear in
    main's source code for the resume path to work.  If someone deletes
    the resume logic by accident, or renames the loop-state key, one of
    these asserts will fire and the commit will have to explicitly
    reconsider the behavior.
    """

    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        import inspect

        from scripts.train.conditional_encdec_sft import main

        return inspect.getsource(main)

    def test_main_reads_resume_from_step(self, main_source: str) -> None:
        """main() must reference ``cfg.resume_from_step`` to branch on resume."""
        assert "resume_from_step" in main_source, (
            "main() no longer references cfg.resume_from_step — the resume "
            "code path is missing"
        )

    def test_main_threads_resuming_flag_to_segment_mode_guard(
        self,
        main_source: str,
    ) -> None:
        """The identity guard receives main()'s real resume state, not a constant."""
        import re

        call = re.search(
            r"_finalize_sft_segment_mode\(\n(?:.*\n)*?        \)", main_source
        )
        assert call is not None, "main() no longer calls _finalize_sft_segment_mode"
        for kwarg in (
            "resuming=resuming,",
            "checkpoint_meta=sft_meta,",
            "checkpoint_dir=sft_checkpoint_dir,",
        ):
            assert kwarg in call.group(0), f"{kwarg} missing from the guard call"

    def test_main_saves_loop_state(self, main_source: str) -> None:
        """main() must save a ``loop_state`` sub-dict in the checkpoint meta."""
        assert '"loop_state"' in main_source, (
            "main() no longer includes a \"loop_state\" key in the saved "
            "checkpoint meta — resume cannot recover training state"
        )

    def test_main_saves_sft_consumed(self, main_source: str) -> None:
        """Keep the rank-0 consumed scalar for progress/reporting metadata."""
        assert "sft_consumed" in main_source, (
            "main() no longer persists sft_consumed progress in loop_state"
        )

    def test_main_selects_and_gathers_rank_local_dataloader_state(
        self, main_source: str
    ) -> None:
        restore = main_source.index("select_dataloader_rank_state(")
        save_predicate = main_source.index("if last_step or")
        gather = main_source.index("build_dataloader_state_envelope(", save_predicate)
        checkpoint = main_source.index("_save_sft_checkpoint(", gather)
        assert restore < save_predicate < gather < checkpoint
        assert '"sft_dataloader_state"' in main_source

    def test_main_commits_dataloader_state_only_after_optimizer_step(
        self, main_source: str
    ) -> None:
        """A failed forward/backward/optimizer step must remain retryable.

        The generator's state describes the batches it has yielded, but that
        state becomes checkpoint-authoritative only after the corresponding
        optimizer update succeeds.
        """
        training_loop = main_source.index("for micro_step in range(grad_accum_steps):")
        batch = main_source.index("next(step_train_gen)", training_loop)
        optimizer_step = main_source.index("optimizer.step()", batch)
        commit = main_source.index(
            "committed_dataloader_state = dict(sft_state.dataloader_state)",
            optimizer_step,
        )
        assert training_loop < batch < optimizer_step < commit

    def test_main_saves_sft_artifact_coverage(self, main_source: str) -> None:
        assert "sft_artifact_coverage" in main_source

    def test_main_load_from_sft_checkpoint_dir(self, main_source: str) -> None:
        """When resuming, main() must load the SFT checkpoint (not the base
        encdec pretrain checkpoint).  The load site must reference
        ``sft_checkpoint_dir``.
        """
        assert "sft_checkpoint_dir" in main_source, (
            "main() no longer references sft_checkpoint_dir — the resume "
            "path cannot find the earlier SFT checkpoint to load from"
        )

    def test_main_passes_rank_to_save_checkpoint(self, main_source: str) -> None:
        """DistMuonAdamW shards optimizer state per rank, so
        ``save_checkpoint`` must be called with ``rank=ddp_rank`` or every
        rank's Muon moments will be lost on a multi-rank resume.  Guard
        against a regression to the pre-resume pattern (``if master_process``
        wrap + default ``rank=0``).
        """
        assert "rank=ddp_rank" in main_source, (
            "main()'s save_checkpoint call must pass rank=ddp_rank so each "
            "DDP rank persists its own sharded optimizer state"
        )
