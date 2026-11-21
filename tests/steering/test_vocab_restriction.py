from __future__ import annotations

from typing import Any

import pytest
import torch as t

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb.vocab_restriction import (
    feature_vocab_allowed_ids,
    restrict_segment_conditions_to_vocab,
    restrict_to_vocab,
)

N_SAE = 64
ALLOWED = (0, 3, 7, 11)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)

# Mirrors tests/nanochat/test_encdec_vocab_checkpoint.py's _build fixture shape — a small,
# known-working config for a real EncoderDecoderGPT.
DEC_KW = {
    "sequence_len": 16,
    "vocab_size": 65536,
    "n_layer": 2,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 64,
    "window_pattern": "L",
}
ENC_KW = {"n_embd": 32, "depth": 2, "n_head": None, "n_kv_head": None, "head_dim": 16}


def _build_model(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    """A real EncoderDecoderGPT with only its buffers materialized.

    Mirrors production's LOAD-path ordering (``to_empty`` -> ``init_buffers``), but skips
    ``init_weights()`` / the checkpoint load: this test only reads ``model.feature_vocab``
    and ``model.encoder.feature_vocab_ids``, neither of which depends on parameter VALUES,
    so there is nothing to buy by materializing (or loading) real weights here.
    """
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(**DEC_KW),  # ty: ignore[invalid-argument-type]
            EncoderConfig(**ENC_KW),  # ty: ignore[invalid-argument-type]
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    m.init_buffers()
    return m


class TestFeatureVocabAllowedIds:
    """Against a REAL EncoderDecoderGPT — proves the helper matches the production shape,
    not just an assumption about its attributes."""

    def test_returns_None_for_an_uncompacted_model(self):
        model = _build_model(None)
        assert feature_vocab_allowed_ids(model) is None

    def test_returns_the_encoders_feature_vocab_ids_buffer_for_a_compacted_model(self):
        model = _build_model(VOCAB)
        ids = feature_vocab_allowed_ids(model)
        assert ids is not None
        assert t.equal(ids, t.tensor(list(ALLOWED), dtype=t.int64))

    def test_returns_the_SAME_buffer_not_a_copy(self):
        # feature_vocab_ids is the checkpoint's own assert_vocab_consistent-verified record;
        # reusing it (not recomputing a fresh tensor) is the point of this helper.
        model = _build_model(VOCAB)
        assert feature_vocab_allowed_ids(model) is model.encoder.feature_vocab_ids


class TestFeatureVocabAllowedIdsDuckTyping:
    """``model`` is duck-typed (mirrors ``TargetModelBundle.model: Any`` in this eval
    layer) — pin the two branches directly against minimal stand-ins, independent of
    EncoderDecoderGPT's own construction machinery."""

    def test_None_branch_on_a_stub(self):
        class _Stub:
            feature_vocab = None

        assert feature_vocab_allowed_ids(_Stub()) is None

    def test_compacted_branch_on_a_stub(self):
        class _Encoder:
            feature_vocab_ids = t.tensor([0, 3], dtype=t.int64)

        class _Stub:
            feature_vocab = VOCAB
            encoder = _Encoder()

        out = feature_vocab_allowed_ids(_Stub())
        assert t.equal(out, t.tensor([0, 3], dtype=t.int64))  # ty: ignore[invalid-argument-type]


class _AllowedDeviceSpy(t.Tensor):
    """A real int64 tensor that records every ``.to(device)`` invocation.

    Because it IS a tensor, ``torch.isin`` accepts it whether or not ``.to`` was called --
    so if ``restrict_to_vocab``'s ``allowed.to(ids.device)`` normalization is removed, the
    function still runs (no crash) yet records no device, and the CPU assertion below flips
    RED. That makes it a CPU-visible mutation trap for the C1 regression the CUDA-only
    cross-device test above cannot catch on a CPU CI run.
    """

    @staticmethod
    def __new__(cls, data: t.Tensor) -> _AllowedDeviceSpy:
        obj = t.Tensor._make_subclass(cls, data)
        obj.recorded_devices = []  # type: ignore[attr-defined]
        obj._plain = data.detach().clone()  # type: ignore[attr-defined]
        return obj

    def to(self, *args: Any, **kwargs: Any) -> t.Tensor:  # type: ignore[override]
        # Record the device restrict_to_vocab normalizes onto, then return a PLAIN moved
        # tensor (drops the spy subclass so downstream membership/indexing is ordinary).
        self.recorded_devices.append(args[0] if args else kwargs.get("device"))  # ty: ignore[unresolved-attribute]
        return self._plain.to(*args, **kwargs)  # ty: ignore[unresolved-attribute]


class TestRestrictToVocab:
    def test_restrict_to_vocab_moves_allowed_onto_ids_device_cpu_trap(self):
        """CPU-visible mutation trap for the C1 device-normalization fix (issue #695
        PR-1): assert, WITHOUT a GPU, that ``restrict_to_vocab`` invokes
        ``allowed.to(ids.device)``. The ``@skipif(not cuda)`` cross-device test below is
        the only other coverage of that line, so on a CPU CI run deleting
        ``allowed.to(ids.device)`` would otherwise pass unnoticed. The spy IS a real
        tensor, so ``torch.isin`` still accepts it if ``.to`` is dropped -- the function
        runs but records no device and this assertion flips RED. Non-vacuous: it also
        pins the kept/dropped split is correct when ``.to`` IS used."""
        ids = t.tensor([0, 3, 5])
        spy = _AllowedDeviceSpy(t.tensor([0, 3]))

        kept, dropped = restrict_to_vocab(ids, spy)

        assert spy.recorded_devices == [ids.device], (  # ty: ignore[unresolved-attribute]
            "restrict_to_vocab must call allowed.to(ids.device) -- the C1 cross-device fix "
            f"(recorded_devices={spy.recorded_devices!r})"  # ty: ignore[unresolved-attribute]
        )
        assert t.equal(kept, t.tensor([0, 3]))
        assert t.equal(dropped, t.tensor([5]))

    def test_kept_and_dropped_split_by_membership(self):
        """Same-device (both default to CPU): the ordinary case, unaffected by
        the device-normalization fix below -- ``allowed.to(ids.device)`` is a
        no-op when the devices already match."""
        ids = t.tensor([0, 3, 5])
        allowed = t.tensor([0, 3])
        kept, dropped = restrict_to_vocab(ids, allowed)
        assert t.equal(kept, t.tensor([0, 3]))
        assert t.equal(dropped, t.tensor([5]))

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_kept_and_dropped_split_by_membership_cross_device(self):
        """C1 regression (issue #695 PR-1 review): ``ids`` on CPU (e.g. a
        freshly built candidate tensor, as in feature_diffmean._select_bag's
        belt-and-suspenders) against ``allowed`` on CUDA (e.g.
        ``model.encoder.feature_vocab_ids`` off a resident encdec model) must
        NOT raise -- ``torch.isin`` requires matching devices, so pre-fix this
        call raised ``RuntimeError: Expected all tensors to be on the same
        device`` on every real (GPU) eval despite being the SUCCESS path (the
        CPU-only test suite -- ``CUDA_VISIBLE_DEVICES=""`` -- never exercised a
        real cross-device pair, so the crash was invisible until a GPU run).
        Skipped (not failed) on this CPU-only env; proves the fix on a GPU box.
        """
        ids = t.tensor([0, 3, 5], device="cpu")
        allowed = t.tensor([0, 3], device="cuda")
        kept, dropped = restrict_to_vocab(ids, allowed)
        assert t.equal(kept.cpu(), t.tensor([0, 3]))
        assert t.equal(dropped.cpu(), t.tensor([5]))
        # kept/dropped stay on ids' device (not allowed's) -- callers index
        # ids-shaped tensors downstream and must not be silently moved to GPU.
        assert kept.device.type == "cpu"
        assert dropped.device.type == "cpu"

    def test_allowed_None_keeps_everything_and_drops_nothing(self):
        ids = t.tensor([9, 1, 4, 100])
        kept, dropped = restrict_to_vocab(ids, None)
        assert t.equal(kept, ids)
        assert dropped.numel() == 0
        assert dropped.dtype == ids.dtype

    def test_kept_preserves_the_ORDER_of_ids_not_of_allowed(self):
        ids = t.tensor([3, 0, 5])  # not ascending
        allowed = t.tensor([0, 3])  # ascending
        kept, dropped = restrict_to_vocab(ids, allowed)
        assert t.equal(kept, t.tensor([3, 0])), "must follow ids' order, not allowed's [0, 3]"
        assert t.equal(dropped, t.tensor([5]))

    def test_nothing_allowed_drops_everything(self):
        ids = t.tensor([1, 2, 3])
        allowed = t.tensor([], dtype=t.int64)
        kept, dropped = restrict_to_vocab(ids, allowed)
        assert kept.numel() == 0
        assert t.equal(dropped, ids)

    def test_empty_ids_returns_two_empty_tensors(self):
        ids = t.tensor([], dtype=t.int64)
        allowed = t.tensor([0, 3])
        kept, dropped = restrict_to_vocab(ids, allowed)
        assert kept.numel() == 0
        assert dropped.numel() == 0

    def test_allowed_None_dropped_tensor_preserves_ids_dtype(self):
        ids = t.tensor([1, 2], dtype=t.int32)
        _, dropped = restrict_to_vocab(ids, None)
        assert dropped.dtype == t.int32


class TestRestrictSegmentConditionsToVocab:
    """Direct unit tests for the shared per-segment drop/warn/raise primitive
    (issue #720 PR-2 Task 2), consumed by ``build_class_margin_members``
    (``gcg_members.py``) and ``build_general_val_doc_member``
    (``general_val_roofline.py``) for their FIXED context bags. Mirrors
    ``teacher_force.py:306-340``'s semantics exactly, but the CALLER owns
    emitting the aggregated ``logger.warning`` -- this returns the dropped
    count instead of logging itself, so it stays reusable across call sites
    with different warning message wording.
    """

    def test_drops_out_of_vocab_ids_keeps_in_vocab_ids(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{1: 10, 99: 20, 2: 30}], allowed,
        )
        assert restricted == [{1: 10, 2: 30}]
        assert total_dropped == 1

    def test_dropped_count_aggregates_across_all_segments(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{1: 1, 50: 1}, {2: 1, 60: 1, 70: 1}], allowed,
        )
        assert restricted == [{1: 1}, {2: 1}]
        assert total_dropped == 3

    def test_no_drop_returns_zero(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{1: 1, 2: 1}], allowed,
        )
        assert restricted == [{1: 1, 2: 1}]
        assert total_dropped == 0

    def test_segment_order_and_count_preserved(self) -> None:
        """3 segments in -> 3 dicts out, SAME order (a dict only ever loses
        keys, never reorders/merges segments). The middle segment is only
        PARTIALLY emptied (id 10 survives alongside dropped id 20), so this
        stays on the non-raising path while still proving order/count."""
        allowed = t.tensor([10, 30], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{10: 0}, {10: 1, 20: 0}, {30: 0}], allowed,
        )
        assert restricted == [{10: 0}, {10: 1}, {30: 0}]
        assert total_dropped == 1

    def test_segment_fully_emptied_by_filter_raises(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            restrict_segment_conditions_to_vocab([{50: 1, 60: 2}], allowed)

    def test_one_of_several_segments_fully_emptied_still_raises(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        with pytest.raises(AxBenchBpbError, match="compacted vocabulary V"):
            restrict_segment_conditions_to_vocab(
                [{1: 1}, {50: 1}], allowed,
            )

    def test_already_empty_segment_does_not_raise(self) -> None:
        """A segment with no ids to begin with was not "emptied BY the
        filter" -- it was already empty. Must not raise."""
        allowed = t.tensor([1, 2], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{}], allowed,
        )
        assert restricted == [{}]
        assert total_dropped == 0

    def test_allowed_ids_none_is_a_strict_noop_returns_input_unchanged(self) -> None:
        """``allowed_ids=None`` must return the SAME object, not a copy --
        the "STRICT no-op" contract. A segment whose ids would ALL be
        dropped under some V must survive completely intact."""
        segment_conditions = [{50: 1, 60: 2}]
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            segment_conditions, None,
        )
        assert restricted is segment_conditions
        assert total_dropped == 0

    def test_allowed_ids_none_never_raises_regardless_of_content(self) -> None:
        restricted, total_dropped = restrict_segment_conditions_to_vocab(
            [{1: 1}, {}, {2: 1, 3: 1}], None,
        )
        assert restricted == [{1: 1}, {}, {2: 1, 3: 1}]
        assert total_dropped == 0

    def test_empty_segment_conditions_list_is_a_noop(self) -> None:
        allowed = t.tensor([1, 2], dtype=t.int64)
        restricted, total_dropped = restrict_segment_conditions_to_vocab([], allowed)
        assert restricted == []
        assert total_dropped == 0
