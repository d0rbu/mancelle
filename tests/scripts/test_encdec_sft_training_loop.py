"""
Tests for encoder-decoder SFT training loop features:
DDP last_step sync, validation evaluation, and wandb logging.
"""

import math
import multiprocessing
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
import torch.distributed as dist
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as _fa_mod
from nanochat.common import DummyWandb
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _SFTValidationLoader,
    _should_run_val_eval,
    create_wandb_run,
    evaluate_encdec_bpb,
    make_sft_data_generator,
    sync_last_step,
)
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder import loss_eval as _shared_loss_eval
from steering.conditional.sampling import Distribution, SamplingBundle, SamplingSelection
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tasks.mixtures import SFTMixture, build_task_mixtures
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.test_encdec_sft_conditioned_integration import (
    _make_boundaries,
    build_synthetic_features,
    make_lazy_patched_sft_generator,
)


class FixedLengthTask(Task):
    """Task returning N conversations, each producing a known token length."""

    def __init__(self, n: int, token_len: int):
        super().__init__()
        self._n = n
        self._token_len = token_len
        self._user_chars = max(1, (token_len - 2) // 2)
        self._asst_chars = max(1, token_len - 2 - self._user_chars)
        # Every index the generator actually pulled, in order.
        self.served: list[int] = []

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        self.served.append(index)
        return {
            "messages": [
                {"role": "user", "content": "u" * self._user_chars},
                {"role": "assistant", "content": "a" * self._asst_chars},
            ]
        }


class SFTMockTokenizer:
    """Minimal tokenizer with render_conversation for SFT generator tests."""

    vocab_size: int = 128

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        user_content = conversation["messages"][0]["content"]
        asst_content = conversation["messages"][1]["content"]
        user_len = len(user_content)
        asst_len = len(asst_content)
        total_len = user_len + asst_len + 2
        ids = [0] + list(range(2, total_len + 1))
        mask = [0] * (user_len + 1) + [1] * (asst_len + 1)
        ids = ids[:total_len]
        mask = mask[:total_len]
        return ids, mask

    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)


def _make_gen(
    task: Task,
    tokenizer: SFTMockTokenizer,
    state: SFTState,
    *,
    B: int = 2,
    T: int = 256,
    single_pass: bool = False,
    sampling_bundle: SamplingBundle | None = None,
    return_seg_doc_id: bool = False,
):
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
            ddp_rank=0,
            ddp_world_size=1,
            buffer_size=100,
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg,
            boundaries=_make_boundaries(),
            sampling_bundle=sampling_bundle,
            min_loss_tokens=1,
            alignment_probe_samples=0,
            single_pass=single_pass,
            return_seg_doc_id=return_seg_doc_id,
            expected_split=SFTSplit.TRAIN,
        ),
    )


def _val_loader(generator, state: SFTState | None = None) -> _SFTValidationLoader:
    """Wrap a raw generator in the type ``evaluate_encdec_bpb`` requires."""
    return _SFTValidationLoader(generator, SFTState() if state is None else state)


@pytest.fixture()
def tokenizer():
    return SFTMockTokenizer()


@pytest.fixture()
def stub_sft_mixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build_task_mixtures tests independent of external dataset caches."""

    def fake_build_mixture(_mixture: object) -> tuple[list[Task], list[Task]]:
        return [FixedLengthTask(n=20, token_len=50)], [FixedLengthTask(n=20, token_len=50)]

    monkeypatch.setattr(
        "tasks.mixtures.build_mixture",
        fake_build_mixture,
    )


class MockEncoderDecoderModel:
    """Returns a constant per-token loss for every position."""

    def __init__(self, loss_value: float, device: str = "cpu"):
        self._loss_value = loss_value
        self._device = device

    def __call__(self, idx, targets, enc_fids, enc_mags, enc_mask, seg_map, loss_reduction="mean"):
        B, T = idx.shape
        per_token = t.full((B, T), self._loss_value, device=self._device)
        if loss_reduction == "mean":
            return per_token.mean()
        return per_token

    def get_device(self):
        return self._device

    def eval(self):
        pass

    def train(self):
        pass


class TestEvaluateEncdecBpb:
    """Tests for evaluate_encdec_bpb function."""

    def test_returns_finite_float(self, tokenizer):
        """evaluate_encdec_bpb returns a finite float."""
        task = FixedLengthTask(n=20, token_len=50)
        state = SFTState()
        gen = _make_gen(task, tokenizer, state, B=2, T=64)
        model = MockEncoderDecoderModel(loss_value=1.0)
        # 1 byte per token for all vocab entries
        token_bytes = t.ones(128, dtype=t.int64)
        result = evaluate_encdec_bpb(
            model, _val_loader(gen, state), steps=2, token_bytes=token_bytes
        )
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_bare_generator_is_rejected(self):
        """The covered-document bound lives on the loader type, not on ducks.

        ``_build_sft_val_loader`` returned a bare generator before PR #1311; a
        refactor that reintroduced one previously reverted validation to
        silent wrap-around with no error at all (review finding 10).

        The rejection is ``@beartype`` enforcing the declared
        ``batches: _SFTValidationLoader`` annotation, not a hand-rolled
        ``isinstance`` block that a later cleanup could delete as redundant
        with the type hint.
        """

        def bare():
            while True:
                yield ()

        with pytest.raises(BeartypeCallHintParamViolation, match=r"batches.*_SFTValidationLoader"):
            evaluate_encdec_bpb(
                MockEncoderDecoderModel(loss_value=1.0),
                bare(),
                steps=1,
                token_bytes=t.ones(128, dtype=t.int64),
            )

    def test_whole_signature_is_enforced_not_just_the_loader(self):
        """``@beartype`` polices EVERY annotation, which an isinstance cannot.

        The deleted hand-rolled block guarded ``batches`` alone; a wrongly
        typed ``steps`` or ``token_bytes`` sailed through into the loop. This
        pins that the decorator — not one bespoke check — is what enforces the
        contract, so the guard cannot regress to covering a single argument.
        """
        state = SFTState()
        state.exhausted = True
        loader = _SFTValidationLoader(iter(()), state)
        with pytest.raises(BeartypeCallHintParamViolation, match=r"steps"):
            evaluate_encdec_bpb(
                MockEncoderDecoderModel(loss_value=1.0),
                loader,
                steps="2",  # ty: ignore[invalid-argument-type]
                token_bytes=t.ones(128, dtype=t.int64),
            )
        with pytest.raises(BeartypeCallHintParamViolation, match=r"token_bytes"):
            evaluate_encdec_bpb(
                MockEncoderDecoderModel(loss_value=1.0),
                loader,
                steps=1,
                token_bytes=[1, 1, 1],  # ty: ignore[invalid-argument-type]
            )

    def test_validation_stops_once_the_generator_reports_exhaustion(self):
        """The eval loop must stop on the generator's own exhaustion flag."""
        state = SFTState()
        state.dataset_size = 2
        yielded = 0

        def batches():
            nonlocal yielded
            while True:
                yielded += 1
                state.consumed += 1
                state.exhausted = state.consumed >= 2
                B, T = 1, 8
                yield (
                    t.zeros(B, T, dtype=t.int32),
                    t.ones(B, T, dtype=t.int64),
                    t.zeros(B, 1, 1, dtype=t.int64),
                    t.zeros(B, 1, 1, dtype=t.int64),
                    t.zeros(B, 1, 1, dtype=t.bool),
                    t.zeros(B, T, dtype=t.int64),
                    None,
                )

        result = evaluate_encdec_bpb(
            MockEncoderDecoderModel(loss_value=1.0),
            _SFTValidationLoader(batches(), state),
            steps=10,
            token_bytes=t.ones(128, dtype=t.int64),
        )

        assert math.isfinite(result)
        assert yielded == 2

    def test_single_pass_validation_never_serves_an_index_twice(self, tokenizer):
        """Validation BPB must be measured over DISTINCT documents.

        With an odd covered count the previous revision checked its bound only
        AFTER a full batch, so the generator wrapped mid-batch and folded a
        duplicated document into ``total_nats`` / ``total_bytes`` before the
        stop fired (review finding 4).
        """
        task = FixedLengthTask(n=5, token_len=20)
        state = SFTState()
        gen = _make_gen(task, tokenizer, state, B=2, T=32, single_pass=True)
        task.served.clear()

        result = evaluate_encdec_bpb(
            MockEncoderDecoderModel(loss_value=1.0),
            _SFTValidationLoader(gen, state),
            steps=50,
            token_bytes=t.ones(128, dtype=t.int64),
        )

        assert math.isfinite(result)
        assert state.exhausted, "the bounded generator must report exhaustion"
        assert sorted(task.served) == [0, 1, 2, 3, 4]
        assert len(task.served) == len(set(task.served)), (
            f"validation re-served documents: {task.served}"
        )

    def test_exhausted_single_pass_generator_repeats_only_rng_neutral_padding(
        self,
        tokenizer,
    ):
        """An early rank can safely keep yielding while peer ranks drain.

        Distributed validation now waits for every rank. The production
        generator contract that makes this safe is stronger than merely "does
        not crash": once exhausted it must not re-serve a document, advance
        stochastic feature-selection state, or contribute a target/feature to
        the global BPB sum.
        """
        task = FixedLengthTask(n=1, token_len=20)
        state = SFTState()
        sampling_rng = t.Generator().manual_seed(1445)
        sampling_bundle = SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
            ),
            percentile_table=t.linspace(0.5, 99.5, 100, dtype=t.float64)
            .unsqueeze(0)
            .expand(200, -1)
            .contiguous(),
            rng=sampling_rng,
        )
        gen = _make_gen(
            task,
            tokenizer,
            state,
            B=2,
            T=32,
            single_pass=True,
            sampling_bundle=sampling_bundle,
            return_seg_doc_id=True,
        )
        task.served.clear()

        first = next(gen)
        assert state.exhausted
        assert bool((first[1] >= 0).any()), "the first batch must contain the real document"
        served_at_exhaustion = list(task.served)
        rng_at_exhaustion = sampling_rng.get_state().clone()

        for _ in range(2):
            batch = next(gen)
            (
                _inputs,
                targets,
                _enc_fids,
                _enc_mags,
                enc_mask,
                seg_map,
                fp_targets,
                seg_doc_id,
            ) = batch
            assert bool((targets == -1).all())
            assert not bool(enc_mask.any())
            assert bool((seg_map == 0).all())
            assert fp_targets is None
            assert bool((seg_doc_id == -1).all())

        assert task.served == served_at_exhaustion == [0]
        assert t.equal(sampling_rng.get_state(), rng_at_exhaustion)

    @pytest.mark.slow
    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_compiled_cuda_exhausted_rank_padding_is_a_finite_zero_bpb_contribution(
        self,
    ):
        """The model must not turn an all-padding rank into NaN BPB nats.

        ``accumulate_bpb_nats_bytes`` masks ignored targets after the forward,
        so a NaN per-token loss would remain NaN under multiplication by zero.
        Exercise both eager and compiled CUDA forwards through the real tiny
        encoder-decoder and require exact zero nats/bytes from the exhausted
        rank payload.
        """
        old_attention_override = _fa_mod._override_impl
        _fa_mod._override_impl = FlashAttnImpl.SDPA
        try:
            model = build_tiny_model(n_layer=2, xattn_frequency=1).to("cuda").eval()
            inputs = t.zeros(2, 64, dtype=t.int32, device="cuda")
            targets = t.full((2, 64), -1, dtype=t.int64, device="cuda")
            enc_fids = t.zeros(2, 1, 1, dtype=t.int64, device="cuda")
            enc_mags = t.zeros(2, 1, 1, dtype=t.int64, device="cuda")
            enc_mask = t.zeros(2, 1, 1, dtype=t.bool, device="cuda")
            seg_map = t.zeros(2, 64, dtype=t.int64, device="cuda")
            args = (inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)

            with t.no_grad():
                eager_loss = model(*args, loss_reduction="none")
                compiled_loss = t.compile(model, dynamic=False)(
                    *args,
                    loss_reduction="none",
                )
            t.cuda.synchronize()

            for loss in (eager_loss, compiled_loss):
                assert bool(t.isfinite(loss).all())
                batch_nats, batch_bytes = _shared_loss_eval.accumulate_bpb_nats_bytes(
                    loss.reshape(-1),
                    targets.reshape(-1),
                    t.ones(128, dtype=t.int64, device="cuda"),
                )
                assert batch_nats.item() == 0.0
                assert batch_bytes.item() == 0
        finally:
            _fa_mod._override_impl = old_attention_override
            t.cuda.empty_cache()

    def test_training_generator_still_wraps(self, tokenizer):
        """``single_pass=False`` keeps multi-epoch training wrap-around."""
        task = FixedLengthTask(n=3, token_len=20)
        state = SFTState()
        gen = _make_gen(task, tokenizer, state, B=2, T=32, single_pass=False)
        task.served.clear()

        for _ in range(4):
            next(gen)

        assert len(task.served) > len(set(task.served))
        assert state.exhausted is False

    def test_known_loss_gives_expected_bpb(self, tokenizer):
        """With uniform loss L=2.0 and 1 byte/token, bpb should be approx L/ln(2)."""
        L = 2.0
        task = FixedLengthTask(n=20, token_len=50)
        state = SFTState()
        gen = _make_gen(task, tokenizer, state, B=2, T=64)
        model = MockEncoderDecoderModel(loss_value=L)
        token_bytes = t.ones(128, dtype=t.int64)
        result = evaluate_encdec_bpb(
            model, _val_loader(gen, state), steps=2, token_bytes=token_bytes
        )
        expected = L / math.log(2)
        assert abs(result - expected) < 0.01, f"Expected bpb ~ {expected:.4f}, got {result:.4f}"

    def test_masked_positions_excluded(self, tokenizer):
        """Completes without error when targets have -1 masked positions."""
        task = FixedLengthTask(n=20, token_len=50)
        state = SFTState()
        gen = _make_gen(task, tokenizer, state, B=2, T=64)
        model = MockEncoderDecoderModel(loss_value=1.0)
        token_bytes = t.ones(128, dtype=t.int64)
        # The generator already masks user tokens with -1, so this should work
        result = evaluate_encdec_bpb(
            model, _val_loader(gen, state), steps=1, token_bytes=token_bytes
        )
        assert math.isfinite(result)

    def test_zero_bytes_returns_inf(self):
        """When all tokens have 0 bytes (all masked), returns inf."""

        def all_masked_gen():
            B, T = 2, 64
            inputs = t.zeros(B, T, dtype=t.int32)
            targets = t.full((B, T), -1, dtype=t.int64)
            enc_fids = t.zeros(B, 1, 1, dtype=t.int64)
            enc_mags = t.zeros(B, 1, 1, dtype=t.int64)
            enc_mask = t.zeros(B, 1, 1, dtype=t.bool)
            seg_map = t.zeros(B, T, dtype=t.int64)
            while True:
                yield inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, None

        model = MockEncoderDecoderModel(loss_value=1.0)
        token_bytes = t.ones(128, dtype=t.int64)
        result = evaluate_encdec_bpb(
            model, _val_loader(all_masked_gen()), steps=1, token_bytes=token_bytes
        )
        assert result == float("inf"), f"Expected inf, got {result}"

    def test_accepts_fp_active_batches_with_seg_doc_id(self):
        """Validation BPB ignores fp metadata but must accept model-safe 8-tuples."""

        def fp_active_gen():
            B, T, n_seg, n_sae = 2, 8, 2, 4
            inputs = t.zeros(B, T, dtype=t.int32)
            targets = t.ones(B, T, dtype=t.int64)
            enc_fids = t.zeros(B, n_seg, 1, dtype=t.int64)
            enc_mags = t.zeros(B, n_seg, 1, dtype=t.int64)
            enc_mask = t.zeros(B, n_seg, 1, dtype=t.bool)
            seg_map = t.zeros(B, T, dtype=t.int64)
            fp_targets = t.zeros(B, n_seg, n_sae, dtype=t.float32)
            seg_doc_id = t.zeros(B, n_seg, dtype=t.int64)
            while True:
                yield (
                    inputs,
                    targets,
                    enc_fids,
                    enc_mags,
                    enc_mask,
                    seg_map,
                    fp_targets,
                    seg_doc_id,
                )

        model = MockEncoderDecoderModel(loss_value=1.0)
        token_bytes = t.ones(128, dtype=t.int64)
        result = evaluate_encdec_bpb(
            model, _val_loader(fp_active_gen()), steps=1, token_bytes=token_bytes
        )
        assert math.isfinite(result)


# ===========================================================================
# Tests: BPB math shared with pretraining (byte-identity golden pin)
# ===========================================================================

# A deliberately awkward fixture: zero-byte vocabulary entries, multi-byte
# entries, ``ignore_index`` targets scattered mid-row, and one row that is
# entirely masked (the shape the ``single_pass`` validation generator pads
# with). Frozen so the golden value below stays meaningful.
_GOLDEN_TOKEN_BYTES = t.tensor([0, 1, 2, 3, 1, 4, 0, 2, 5, 1, 3, 2, 1, 1, 7, 0], dtype=t.int64)
_GOLDEN_TARGETS = (
    t.tensor([[1, 2, -1, 4, 5], [6, 7, 8, -1, -1]], dtype=t.int64),
    t.tensor([[0, 0, 3, 3, 9], [-1, -1, -1, -1, -1]], dtype=t.int64),
)
_GOLDEN_LOSSES = (
    t.arange(10, dtype=t.float32).reshape(2, 5) * 0.3125 + 0.125,
    t.arange(10, dtype=t.float32).reshape(2, 5) * -0.0625 + 2.5,
)
# ``float.hex()`` of the value the PRE-DRY inline arithmetic produced on this
# fixture, captured on origin/main before the helpers were wired in. Pinned as
# a hex float so the comparison is bit-exact rather than decimal-rounded: a
# silent change in eval numerics would invalidate every ``min_val_bpb``
# comparison ever recorded, so "close enough" is not a passing grade.
_GOLDEN_VAL_BPB_HEX = "0x1.de733c30fa3e0p-1"  # 0.9344729242121694 over 22 bytes


def _reference_inline_bpb(losses, targets_list, token_bytes) -> float:
    """The exact inline arithmetic ``evaluate_encdec_bpb`` carried pre-DRY.

    Frozen here on purpose. If the shared helpers ever diverge from what SFT
    used to compute, this reference — not a re-derivation of the helpers —
    is what catches it.
    """
    total_nats = t.tensor(0.0, dtype=t.float32)
    total_bytes = t.tensor(0, dtype=t.int64)
    for loss2d, targets in zip(losses, targets_list, strict=True):
        flat_loss = loss2d.reshape(-1)
        y = targets.reshape(-1)
        valid = y >= 0
        y_safe = t.where(valid, y, t.zeros_like(y))
        num_bytes2d = t.where(
            valid,
            token_bytes[y_safe],
            t.zeros_like(y, dtype=token_bytes.dtype),
        )
        total_nats += (flat_loss * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    total_nats_val = total_nats.item()
    total_bytes_val = total_bytes.item()
    if total_bytes_val == 0:
        return float("inf")
    return total_nats_val / (math.log(2) * total_bytes_val)


class _ScriptedLossModel:
    """Returns a scripted per-token loss tensor for each successive batch."""

    def __init__(self, losses):
        self._losses = list(losses)
        self._calls = 0

    def __call__(self, idx, targets, enc_fids, enc_mags, enc_mask, seg_map, loss_reduction="mean"):
        assert loss_reduction == "none"
        loss = self._losses[self._calls]
        self._calls += 1
        return loss

    def get_device(self):
        return "cpu"


def _golden_loader(targets_list):
    state = SFTState()
    remaining = len(targets_list)

    def batches():
        nonlocal remaining
        for targets in targets_list:
            remaining -= 1
            state.exhausted = remaining == 0
            B, T = targets.shape
            yield (
                t.zeros(B, T, dtype=t.int32),
                targets,
                t.zeros(B, 1, 1, dtype=t.int64),
                t.zeros(B, 1, 1, dtype=t.int64),
                t.zeros(B, 1, 1, dtype=t.bool),
                t.zeros(B, T, dtype=t.int64),
                None,
            )

    return _SFTValidationLoader(batches(), state)


class TestBpbMathSharedWithPretraining:
    """``val/bpb`` must be byte-identical before and after the DRY."""

    def test_matches_the_frozen_inline_reference_exactly(self):
        result = evaluate_encdec_bpb(
            _ScriptedLossModel(_GOLDEN_LOSSES),
            _golden_loader(_GOLDEN_TARGETS),
            steps=len(_GOLDEN_TARGETS),
            token_bytes=_GOLDEN_TOKEN_BYTES,
        )
        expected = _reference_inline_bpb(_GOLDEN_LOSSES, _GOLDEN_TARGETS, _GOLDEN_TOKEN_BYTES)
        assert result == expected, (
            f"BPB drifted from the pre-DRY arithmetic: {result!r} != {expected!r}"
        )

    def test_golden_value_is_pinned_bit_exactly(self):
        """A literal captured on origin/main, independent of the reference above."""
        result = evaluate_encdec_bpb(
            _ScriptedLossModel(_GOLDEN_LOSSES),
            _golden_loader(_GOLDEN_TARGETS),
            steps=len(_GOLDEN_TARGETS),
            token_bytes=_GOLDEN_TOKEN_BYTES,
        )
        assert float(result).hex() == _GOLDEN_VAL_BPB_HEX, (
            f"val/bpb changed: {float(result).hex()} != {_GOLDEN_VAL_BPB_HEX} "
            f"({result!r} vs {float.fromhex(_GOLDEN_VAL_BPB_HEX)!r})"
        )

    def test_actually_calls_the_shared_helpers(self, monkeypatch):
        """Non-vacuity: equality alone cannot tell DRY from a copy-paste.

        Each conjunct gets its own channel — this one proves the two shared
        functions are on the call path, the two above prove the number did not
        move.
        """
        calls: list[str] = []
        real_accumulate = _shared_loss_eval.accumulate_bpb_nats_bytes
        real_reduce = _shared_loss_eval.reduce_bpb

        def spy_accumulate(*args, **kwargs):
            calls.append("accumulate")
            return real_accumulate(*args, **kwargs)

        def spy_reduce(*args, **kwargs):
            calls.append("reduce")
            return real_reduce(*args, **kwargs)

        monkeypatch.setattr(
            "scripts.train.conditional_encdec_sft.accumulate_bpb_nats_bytes",
            spy_accumulate,
        )
        monkeypatch.setattr("scripts.train.conditional_encdec_sft.reduce_bpb", spy_reduce)
        evaluate_encdec_bpb(
            _ScriptedLossModel(_GOLDEN_LOSSES),
            _golden_loader(_GOLDEN_TARGETS),
            steps=len(_GOLDEN_TARGETS),
            token_bytes=_GOLDEN_TOKEN_BYTES,
        )
        assert calls == ["accumulate", "accumulate", "reduce"], calls

    def test_zero_total_bytes_is_inf_on_both_sides(self):
        """The degenerate-input edge case ``reduce_bpb`` broadens to ``<= 0``."""
        zero_bytes = t.zeros(16, dtype=t.int64)
        result = evaluate_encdec_bpb(
            _ScriptedLossModel(_GOLDEN_LOSSES),
            _golden_loader(_GOLDEN_TARGETS),
            steps=len(_GOLDEN_TARGETS),
            token_bytes=zero_bytes,
        )
        assert result == float("inf")
        assert _reference_inline_bpb(_GOLDEN_LOSSES, _GOLDEN_TARGETS, zero_bytes) == result

    def test_fully_masked_batch_contributes_nothing(self):
        """An all-``ignore_index`` row (the single_pass pad shape) is a no-op.

        Scoring the padded batch and not scoring it must give the same number,
        which is the property the ``targets_flat.int() < 0`` branch in
        ``accumulate_bpb_nats_bytes`` has to preserve.
        """
        masked = t.full((2, 5), -1, dtype=t.int64)
        padded_targets = (*_GOLDEN_TARGETS, masked)
        padded_losses = (*_GOLDEN_LOSSES, t.full((2, 5), 9.0, dtype=t.float32))
        with_pad = evaluate_encdec_bpb(
            _ScriptedLossModel(padded_losses),
            _golden_loader(padded_targets),
            steps=len(padded_targets),
            token_bytes=_GOLDEN_TOKEN_BYTES,
        )
        assert float(with_pad).hex() == _GOLDEN_VAL_BPB_HEX

    def test_zero_steps_yields_inf(self):
        """Empty accumulation: no batch, no bytes, ``inf`` (not a ZeroDivision)."""
        result = evaluate_encdec_bpb(
            _ScriptedLossModel(()),
            _golden_loader(()),
            steps=0,
            token_bytes=_GOLDEN_TOKEN_BYTES,
        )
        assert result == float("inf")


# ===========================================================================
# Tests: real 2-rank validation eval (DDP collective alignment)
# ===========================================================================


def _ddp_eval_worker(
    rank: int,
    world_size: int,
    init_file: str,
    steps: int,
    queue,
) -> None:
    """Run ``evaluate_encdec_bpb`` on one real gloo rank.

    Each rank's generator drains its covered slice at a DIFFERENT batch, which
    is exactly what best-fit packing does in production: a rank-local ``break``
    then leaves the loop's iteration count rank-dependent, and the ranks
    desynchronise on the very next collective.
    """
    import torch as t
    import torch.distributed as dist

    from scripts.train.conditional_encdec_sft import (
        SFTState,
        _SFTValidationLoader,
        evaluate_encdec_bpb,
    )

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        covered_docs = 1 if rank == 0 else 4
        state = SFTState()
        state.dataset_size = covered_docs
        yielded = 0

        class RankLossModel:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(
                self,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                loss_reduction="mean",
            ):
                assert loss_reduction == "none"
                self.calls += 1
                # Rank 0's sole real batch has loss 1. Rank 1's four real
                # batches have losses 2, 3, 4, 5, making every dropped tail
                # batch observable in the final pooled BPB.
                loss_value = 1.0 if rank == 0 else float(self.calls + 1)
                return t.full(idx.shape, loss_value, dtype=t.float32)

            def get_device(self):
                return "cpu"

        model = RankLossModel()

        def batches():
            nonlocal yielded
            while True:
                yielded += 1
                if state.exhausted:
                    targets = t.full((1, 8), -1, dtype=t.int64)
                else:
                    state.consumed += 1
                    targets = t.ones(1, 8, dtype=t.int64)
                state.exhausted = state.consumed >= covered_docs
                yield (
                    t.zeros(1, 8, dtype=t.int32),
                    targets,
                    t.zeros(1, 1, 1, dtype=t.int64),
                    t.zeros(1, 1, 1, dtype=t.int64),
                    t.zeros(1, 1, 1, dtype=t.bool),
                    t.zeros(1, 8, dtype=t.int64),
                    None,
                )

        bpb = evaluate_encdec_bpb(
            model,
            _SFTValidationLoader(batches(), state),
            steps=steps,
            token_bytes=t.ones(128, dtype=t.int64),
        )
        queue.put(("ok", rank, float(bpb), yielded, model.calls, state.consumed))
    except BaseException as exc:  # noqa: BLE001 - report anything to the parent
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}", -1, -1, -1))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    ("steps", "expected_iterations", "expected_rank1_real_batches", "expected_mean_loss"),
    [
        pytest.param(6, 4, 4, 3.0, id="all-ranks-exhausted"),
        pytest.param(2, 2, 2, 2.0, id="eval-step-budget"),
    ],
)
def test_validation_eval_keeps_two_real_ranks_in_collective_lockstep(
    steps: int,
    expected_iterations: int,
    expected_rank1_real_batches: int,
    expected_mean_loss: float,
):
    """Two real gloo ranks retain every in-budget tail batch exactly once.

    Rank 0 exhausts after one real batch and then contributes fully masked
    padding while rank 1 contributes four distinguishable real batches. The
    first case proves collective termination waits for the slowest rank and
    matches the hand-computed pooled oracle; the second proves ``steps`` still
    caps evaluation even when a rank remains non-exhausted.
    """
    world_size = 2
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = str(Path(tmp_dir) / "gloo_rendezvous")
        processes = [
            context.Process(
                target=_ddp_eval_worker,
                args=(rank, world_size, init_file, steps, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        try:
            results = [queue.get(timeout=120) for _ in range(world_size)]
        except Exception as exc:  # queue.Empty => a rank hung on a collective
            for process in processes:
                process.kill()
            raise AssertionError(
                "a validation-eval rank never finished; the ranks desynchronised on a collective"
            ) from exc
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)

    failures = [result for result in results if result[0] != "ok"]
    assert not failures, f"rank failures: {failures}"
    exit_codes = [process.exitcode for process in processes]
    assert exit_codes == [0, 0], f"non-zero rank exit codes: {exit_codes}"
    by_rank = {
        rank: (value, yielded, calls, consumed)
        for _status, rank, value, yielded, calls, consumed in results
    }
    assert set(by_rank) == {0, 1}
    expected_bpb = expected_mean_loss / math.log(2)
    for rank, (bpb, yielded, calls, _consumed) in by_rank.items():
        assert bpb == pytest.approx(expected_bpb), (rank, bpb, expected_bpb)
        assert yielded == calls == expected_iterations
    assert by_rank[0][3] == 1
    assert by_rank[1][3] == expected_rank1_real_batches


# ===========================================================================
# Tests: rank-synchronized startup preflight
# ===========================================================================


def _preflight_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    """One real gloo rank running a startup preflight that fails on rank 1 only.

    This is the shape of every rank-local startup raise: a shared or flaky
    filesystem lets exactly one rank fail to read a shard or a statistics
    artifact. Without the collective, that rank raises and exits while its
    peers walk into the next collective (here a plain ``all_reduce``, in
    production the first ``sync_last_step`` or loss reduction) and block until
    the NCCL watchdog kills the job.
    """
    import torch as t
    import torch.distributed as dist

    from errors import DataLoadError
    from steering._common.ddp import AlignedRankFailure, rank_synchronized_stage

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        try:
            with rank_synchronized_stage("percentile coverage", device="cpu"):
                if rank == 1:
                    raise DataLoadError("feature statistics artifact unreadable on this rank")
            # Only reachable if the preflight let this rank through: enter the
            # collective the failed peer will never arrive at.
            dist.all_reduce(t.zeros(1))
            queue.put(("survived", rank, "preflight did not abort this rank"))
        except AlignedRankFailure as exc:
            queue.put(("preflight-failure", rank, str(exc)))
    except BaseException as exc:  # noqa: BLE001 - report anything to the parent
        queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_one_rank_failing_preflight_aborts_every_rank():
    """A rank-local startup failure must not leave its peers hanging.

    Before the collective, rank 1 raised ``DataLoadError`` locally and exited
    while rank 0 blocked in ``all_reduce`` forever — the run died on the NCCL
    watchdog many minutes later with an error naming nothing useful. Both ranks
    must now fail, and both messages must name the rank that failed and why.
    """
    world_size = 2
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = str(Path(tmp_dir) / "gloo_preflight_rendezvous")
        processes = [
            context.Process(
                target=_preflight_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        try:
            results = [queue.get(timeout=120) for _ in range(world_size)]
        except Exception as exc:  # queue.Empty => the healthy rank hung
            for process in processes:
                process.kill()
            raise AssertionError(
                "a rank never finished: the healthy rank blocked on a collective "
                "its failed peer never reached"
            ) from exc
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)

    by_rank = {rank: (status, message) for status, rank, message in results}
    assert set(by_rank) == {0, 1}, by_rank
    for rank, (status, message) in by_rank.items():
        assert status == "preflight-failure", (rank, status, message)
        assert "'percentile coverage'" in message, message
        assert "rank 1" in message, message
        assert "feature statistics artifact unreadable" in message, message
    assert "This rank passed the stage" in by_rank[0][1], by_rank[0][1]
    assert "failed on rank 1 (first failing rank 1)" in by_rank[1][1], by_rank[1][1]


class TestPreflightSyncSingleProcess:
    """Single-process behaviour: no process group, no collective, same raise."""

    def test_local_error_propagates_unwrapped_without_ddp(self):
        with pytest.raises(ValueError, match=r"local boom"):
            with rank_synchronized_stage("statistics coverage", device="cpu"):
                raise ValueError("local boom")

    def test_success_is_transparent(self):
        marker = []
        with rank_synchronized_stage("statistics coverage", device="cpu"):
            marker.append(1)
        assert marker == [1]


# ===========================================================================
# Tests: DDP last_step synchronization
# ===========================================================================


class TestSyncLastStep:
    """Tests for the DDP last_step synchronization function."""

    def test_no_ddp_returns_input_unchanged(self):
        """When ddp=False, sync_last_step should return the input as-is."""
        assert sync_last_step(True, "cpu", ddp=False) is True
        assert sync_last_step(False, "cpu", ddp=False) is False

    @pytest.mark.parametrize(
        ("local_exhausted", "remote_exhausted", "expected"),
        [
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ],
    )
    def test_ddp_stops_only_when_every_rank_is_exhausted(
        self,
        local_exhausted: bool,
        remote_exhausted: bool,
        expected: bool,
    ):
        """Pin the complete two-rank Boolean contract and the MIN collective."""
        observed_ops = []

        def fake_all_reduce(tensor, op=None):
            observed_ops.append(op)
            tensor.fill_(min(int(tensor.item()), int(remote_exhausted)))

        with patch(
            "scripts.train.conditional_encdec_sft.dist.all_reduce",
            side_effect=fake_all_reduce,
        ):
            result = sync_last_step(local_exhausted, "cpu", ddp=True)

        assert result is expected
        assert observed_ops == [dist.ReduceOp.MIN]


# ===========================================================================
# Tests: Validation data loading
# ===========================================================================


class TestBuildSftMixture:
    """build_task_mixtures returns the train and validation datasets."""

    def test_returns_train_and_val(self, stub_sft_mixture: None):
        """Should return a (train_mixture, val_mixture) tuple with val data for SAFETY."""
        train, val = build_task_mixtures(SFTMixture.SAFETY)
        assert len(train) > 0, "Train mixture should have examples"
        assert val is not None, "SAFETY mixture should have val data"
        assert len(val) > 0, "Val mixture should have examples"

    def test_val_data_generates_valid_batches(self, tokenizer, stub_sft_mixture: None):
        """Val generator should produce the same 7-tuple shape as train."""
        _, val = build_task_mixtures(SFTMixture.SAFETY)
        assert val is not None
        state = SFTState()
        gen = _make_gen(val, tokenizer, state, B=2, T=128)
        batch = next(gen)
        assert len(batch) == 7, f"Expected 7-tuple, got {len(batch)}-tuple"
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
        assert inputs.shape == (2, 128)
        assert targets.shape == (2, 128)


# ===========================================================================
# Tests: Validation evaluation trigger logic
# ===========================================================================


class TestEvalTrigger:
    """Eval should run at last_step and every eval_every steps."""

    def test_eval_triggers_at_last_step(self):
        assert _should_run_val_eval(has_val=True, eval_every=150, step=7, last_step=True)

    def test_eval_triggers_at_eval_every(self):
        assert _should_run_val_eval(has_val=True, eval_every=150, step=150, last_step=False)

    def test_eval_skips_non_matching_steps(self):
        assert not _should_run_val_eval(has_val=True, eval_every=150, step=42, last_step=False)

    def test_eval_disabled_when_eval_every_zero(self):
        assert not _should_run_val_eval(has_val=True, eval_every=0, step=150, last_step=False)

    def test_eval_every_zero_suppresses_even_the_last_step_eval(self):
        assert not _should_run_val_eval(has_val=True, eval_every=0, step=99, last_step=True)

    def test_eval_at_step_zero(self):
        assert _should_run_val_eval(has_val=True, eval_every=150, step=0, last_step=False)


class TestMinValBpbTracking:
    """min_val_bpb should track the minimum validation BPB seen."""

    def test_tracks_minimum(self):
        min_val_bpb = float("inf")
        for bpb in [3.5, 2.1, 2.8, 1.9, 2.3]:
            if bpb < min_val_bpb:
                min_val_bpb = bpb
        assert min_val_bpb == 1.9

    def test_starts_at_inf(self):
        min_val_bpb = float("inf")
        assert min_val_bpb == float("inf")


# ===========================================================================
# Tests: Wandb logging
# ===========================================================================


class TestWandbInit:
    """Wandb should use DummyWandb for non-master or dummy run name."""

    def test_dummy_wandb_for_non_master(self):
        """Non-master ranks should get DummyWandb."""
        run = create_wandb_run(run_name="test", master_process=False, user_config={})
        assert isinstance(run, DummyWandb)

    def test_dummy_wandb_for_dummy_run_name(self):
        """run_name='dummy' should get DummyWandb regardless of rank."""
        run = create_wandb_run(run_name="dummy", master_process=True, user_config={})
        assert isinstance(run, DummyWandb)

    def test_dummy_wandb_log_is_noop(self):
        """DummyWandb.log() should not raise."""
        run = DummyWandb()
        run.log({"train/loss": 1.0, "step": 1})

    def test_dummy_wandb_finish_is_noop(self):
        """DummyWandb.finish() should not raise."""
        run = DummyWandb()
        run.finish()


class TestWandbTrainMetrics:
    """Train metrics should have the expected keys."""

    def test_train_metric_keys(self):
        expected_keys = {
            "step",
            "total_training_time",
            "train/loss",
            "train/lrm",
            "train/dt",
        }
        metrics = {
            "step": 10,
            "total_training_time": 5.0,
            "train/loss": 2.5,
            "train/lrm": 1.0,
            "train/dt": 0.5,
        }
        assert expected_keys == set(metrics.keys())

    def test_val_metric_keys(self):
        expected_keys = {"step", "total_training_time", "val/bpb"}
        metrics = {
            "step": 150,
            "total_training_time": 60.0,
            "val/bpb": 2.1,
        }
        assert expected_keys == set(metrics.keys())
