from __future__ import annotations

import pytest

from errors import EvalConditioningError
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_vocab import FeatureVocab

N_SAE = 64
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=(0, 3, 7, 11))  # compacted: V=4 < 64
CKPT = "/fake/ckpt/dir"


class TestDeferred:
    def test_deferred_on_a_LEGACY_checkpoint_is_a_no_op(self):
        EvalConditioning.deferred("base_eval").assert_loadable(None, checkpoint_dir=CKPT)

    def test_deferred_on_a_COMPACTED_checkpoint_raises_NotImplementedError(self):
        with pytest.raises(NotImplementedError) as exc:
            EvalConditioning.deferred("base_eval").assert_loadable(VOCAB, checkpoint_dir=CKPT)
        msg = str(exc.value)
        assert "base_eval" in msg, "the message must name the eval"
        assert "dump_feature_mask" in msg, "must tell the operator how to recover the mask"
        assert CKPT in msg, "must name the checkpoint"

    def test_deferred_message_does_NOT_advise_nulling_the_band(self):
        msg = ""
        try:
            EvalConditioning.deferred("x").assert_loadable(VOCAB, checkpoint_dir=CKPT)
        except NotImplementedError as e:
            msg = str(e).lower()
        assert "freq_min" not in msg and "freq_max" not in msg
        assert "null" not in msg and "disable" not in msg  # owner rule 1

    def test_deferred_requires_a_nonempty_name(self):
        with pytest.raises(EvalConditioningError):
            EvalConditioning.deferred("")


class TestNotEvaluating:
    def test_not_evaluating_reaching_an_eval_load_RAISES_regardless_of_compaction(self):
        # assert_loadable is only called on the EVAL path, so reaching it with not_evaluating()
        # is always a mis-declaration — legacy AND compacted.
        for vocab in (None, VOCAB):
            with pytest.raises(EvalConditioningError, match="not_evaluating"):
                EvalConditioning.not_evaluating().assert_loadable(vocab, checkpoint_dir=CKPT)


class TestConditionsWithinVocab:
    """PR-1 Task 1: a PERMIT stance — unlike deferred(), it lets a compacted checkpoint
    load, because the eval restricts its OWN feature selection to V downstream (the
    stance cannot verify that at load time; see steering/eval/axbench_bpb/vocab_restriction.py).
    """

    def test_permits_a_COMPACTED_checkpoint_no_raise(self):
        EvalConditioning.conditions_within_vocab("axbench_bpb").assert_loadable(
            VOCAB, checkpoint_dir=CKPT
        )

    def test_permits_a_LEGACY_checkpoint_no_raise(self):
        EvalConditioning.conditions_within_vocab("axbench_bpb").assert_loadable(
            None, checkpoint_dir=CKPT
        )

    def test_requires_a_nonempty_name(self):
        with pytest.raises(EvalConditioningError):
            EvalConditioning.conditions_within_vocab("")

    def test_deferred_on_a_COMPACTED_checkpoint_STILL_raises_after_adding_this_stance(self):
        # Non-vacuity / mutation-check (Task 1 Step 5): adding conditions_within_vocab must
        # NOT weaken the PRE-EXISTING deferred() contract. A dedicated, self-contained pin for
        # THIS change, in addition to TestDeferred's own coverage above.
        with pytest.raises(NotImplementedError):
            EvalConditioning.deferred("x").assert_loadable(VOCAB, checkpoint_dir=CKPT)


class TestType:
    def test_is_frozen(self):
        c = EvalConditioning.not_evaluating()
        with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError subclasses this
            c._stance = None  # frozen dataclass  # ty: ignore[invalid-assignment]
