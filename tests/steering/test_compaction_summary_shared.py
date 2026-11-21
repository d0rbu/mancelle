"""ONE compaction summary, three callers (#695 review finding 14).

``format_compaction_summary`` lived in ``scripts/train/conditional_encdec_train.py`` — a
torchrun entrypoint — so the two *other* programs that build a compacted encoder could
not import it, and each hand-rolled the string instead. The three copies had already
drifted, and not cosmetically: **only the training one printed the feature-indexed
parameter saving**, i.e. the number that decides whether the run fits in VRAM. SFT and
conversion operators never saw it.

Pinned here: one implementation, in an importable home, called by all three, and no
caller left hand-rolling its own.

The provenance clause is a *parameter*, not a fork, because the three differ in kind and
not merely in wording: the trainer and the converter RESOLVE the vocabulary from
``cfg.feature_mask``, while SFT INHERITS it from the base checkpoint and can neither
create nor widen one. "no feature_mask is configured" is therefore an outright lie in an
SFT run that configures a mask against a legacy (uncompacted) checkpoint — so SFT must be
able to say what is actually true of it.
"""

from __future__ import annotations

import ast
import inspect

import pytest
from omegaconf import DictConfig, OmegaConf

import scripts.preprocess.convert_decoder_to_encdec as convert
import scripts.train.conditional_encdec_sft as sft
import scripts.train.conditional_encdec_train as train
from scripts.train.conditional_encdec_train import build_encdec_meta_model_from_training_cfg
from steering.conditional.compaction_summary import (
    format_compaction_summary,
    no_compaction_reason,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)

# Every caller must state WHY there is no vocabulary — the model cannot know (see
# TestTheAbsenceReasonIsNotAFlatLie). Tests that are not about the reason use this.
_REASON = "no feature_mask is configured"


def _cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "depth": 2,
            "aspect_ratio": 32,
            "head_dim": 32,
            "max_seq_len": 16,
            "window_pattern": "L",
            "num_buckets": 4,
            "encoder": {
                "n_embd": 32,
                "depth": 2,
                "n_head": None,
                "n_kv_head": None,
                "head_dim": 16,
            },
            "cross_attention": {"frequency": 1, "n_head": None, "n_kv_head": None},
            "feature_prediction": None,
        }
    )


def _main_of(module) -> ast.FunctionDef:
    """The module's ``main()`` AST. Both bodies need a Hydra config, a real checkpoint and
    DDP to run, so — exactly as ``test_encdec_sft_vocab.TestSftWiring`` already does — the
    wiring is proven by reading them."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError(f"{module.__name__} has no main()")


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == name)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        )
    ]


@pytest.fixture(scope="module")
def model():
    return build_encdec_meta_model_from_training_cfg(
        _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=VOCAB
    )


@pytest.fixture(scope="module")
def full_width_model():
    return build_encdec_meta_model_from_training_cfg(
        _cfg(), base_vocab=128, n_sae_features=N_SAE, feature_vocab=None
    )


class TestOneSharedImplementation:
    def test_the_summary_lives_somewhere_importable(self):
        """Not in a torchrun entrypoint. That is the whole reason it got copied twice."""
        assert format_compaction_summary.__module__ == "steering.conditional.compaction_summary"

    def test_all_three_callers_resolve_to_the_SAME_function_object(self):
        assert train.format_compaction_summary is format_compaction_summary
        assert sft.format_compaction_summary is format_compaction_summary
        assert convert.format_compaction_summary is format_compaction_summary


class TestNoCallerHandRollsItsOwn:
    @pytest.mark.parametrize("module", [sft, convert], ids=["sft", "convert"])
    def test_main_calls_the_shared_summary(self, module):
        assert _calls_named(_main_of(module), "format_compaction_summary"), (
            f"{module.__name__}.main() does not call format_compaction_summary — it is "
            "still hand-rolling the compaction log, so its operator never sees the "
            "feature-indexed parameter saving."
        )

    @pytest.mark.parametrize("module", [sft, convert], ids=["sft", "convert"])
    def test_main_hand_rolls_no_summary_string_of_its_own(self, module):
        body = ast.unparse(_main_of(module))
        assert "COMPACTED to" not in body, (
            f"{module.__name__}.main() still contains a hand-rolled "
            "'Encoder feature dictionary: COMPACTED to ...' string. There must be exactly "
            "one implementation."
        )


class TestTheParamSavingReachesEveryOperator:
    """The drift that mattered: only the trainer printed the number that decides VRAM fit."""

    def test_summary_quotes_the_feature_indexed_param_saving(self, model):
        msg = format_compaction_summary(model, no_vocab_reason=_REASON)

        enc = model.encoder.config
        kv_dim = enc.n_kv_head * enc.head_dim
        saved = (N_SAE - len(ALLOWED)) * (enc.n_embd + len(model.encoder.value_embeds) * kv_dim)
        assert f"{saved:,}" in msg
        assert "Feature-indexed encoder params" in msg

    def test_the_provenance_clause_does_not_cost_the_param_saving(self, model):
        """SFT/convert pass a note; they must still get the rich body."""
        msg = format_compaction_summary(
            model,
            vocab_source="inherited from the base checkpoint",
            no_vocab_reason=_REASON,
        )
        assert "Feature-indexed encoder params" in msg
        assert "inherited from the base checkpoint" in msg


class TestProvenanceIsTruthfulPerCaller:
    def test_the_trainers_reason_names_the_config(self, full_width_model):
        msg = format_compaction_summary(
            full_width_model, no_vocab_reason=no_compaction_reason(None, N_SAE)
        )
        assert "no compaction" in msg.lower()
        assert "no feature_mask is configured" in msg

    def test_sft_can_say_the_CHECKPOINT_is_why_there_is_no_compaction(self, full_width_model):
        """An SFT run may configure a feature_mask against a LEGACY checkpoint. It still
        gets a full-width encoder — because the checkpoint is not compacted, NOT because
        no mask is configured. The trainer's reason would be a lie there."""
        msg = format_compaction_summary(
            full_width_model, no_vocab_reason="the base checkpoint is not compacted"
        )
        assert "the base checkpoint is not compacted" in msg
        assert "no feature_mask is configured" not in msg


class TestTheAbsenceReasonIsNotAFlatLie:
    """FINDING 9. ``from_mask`` returns ``None`` for TWO worlds, not one.

    ``FeatureVocab.from_mask`` normalises an IDENTITY vocabulary — a mask whose allowed
    universe is the whole dictionary — away to ``None``, exactly as it does for "no mask
    at all". So ``feature_vocab is None`` cannot distinguish:

      (a) no ``feature_mask`` is configured, and
      (b) one IS configured, and it allows every feature.

    The summary defaulted to the flat claim "no feature_mask is configured". In case (b)
    that is told to an operator who is *looking at their mask in the config* — a
    confusing lie, and one that would send them hunting for a key they can see is set.
    """

    IDENTITY = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(N_SAE)))

    def test_the_identity_mask_really_does_normalise_to_None(self):
        """The premise. If this ever stops holding, the rest of the class is moot."""
        assert FeatureVocab.from_mask(self.IDENTITY, N_SAE) is None
        assert FeatureVocab.from_mask(None, N_SAE) is None

    def test_an_identity_mask_is_NOT_reported_as_no_mask_configured(self, full_width_model):
        msg = format_compaction_summary(
            full_width_model, no_vocab_reason=no_compaction_reason(self.IDENTITY, N_SAE)
        )
        assert "no feature_mask is configured" not in msg, (
            "a feature_mask IS configured — it just allows everything. Telling the "
            "operator none is configured is a flat lie about their own config."
        )

    def test_it_says_what_is_ACTUALLY_true_of_the_identity_mask(self, full_width_model):
        msg = format_compaction_summary(
            full_width_model, no_vocab_reason=no_compaction_reason(self.IDENTITY, N_SAE)
        )
        assert "no compaction" in msg.lower()
        assert "feature_mask" in msg, "must name the mask that IS configured"
        assert "whitelist" in msg, "must name its mode so the operator can find it"
        assert "all" in msg.lower(), "must say the mask allows every feature — that is WHY"

    def test_the_two_worlds_produce_DIFFERENT_reasons(self):
        assert no_compaction_reason(None, N_SAE) != no_compaction_reason(self.IDENTITY, N_SAE)

    def test_a_real_mask_that_prunes_never_reaches_the_no_vocab_branch(self):
        """Sanity: a pruning mask yields a vocab, so no reason is used at all."""
        pruning = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED)
        assert FeatureVocab.from_mask(pruning, N_SAE) == VOCAB


class TestTheSignatureCannotCarryADisagreeingPair:
    """FINDING 14. Three arguments where one sufficed, and nothing checked they agreed.

    ``feature_vocab`` and ``n_sae_features`` are both derivable from ``model`` — the
    former is ``model.feature_vocab``, the latter ``model.encoder.config.n_sae_features``
    — and ``EncoderDecoderGPT.__init__`` already refuses to build a model whose vocab
    disagrees with its width. Passing them alongside the model let a caller hand in a
    vocabulary the model does not have, and the summary would happily print the RATIO OF
    A MODEL THAT DOES NOT EXIST. Now they are read off the model and the pair is
    unrepresentable.
    """

    def test_the_derivable_arguments_are_GONE(self):
        params = inspect.signature(format_compaction_summary).parameters
        assert "feature_vocab" not in params
        assert "n_sae_features" not in params
        positional = [
            p.name
            for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == ["model"], f"one argument suffices; got {positional}"

    def test_the_summary_reads_the_vocab_and_the_width_OFF_the_model(self, model):
        msg = format_compaction_summary(model, no_vocab_reason=_REASON)
        assert f"{len(ALLOWED):,}/{N_SAE:,}" in msg

    def test_a_full_width_model_takes_the_no_vocab_branch_with_no_help(self, full_width_model):
        msg = format_compaction_summary(full_width_model, no_vocab_reason=_REASON)
        assert "no compaction" in msg.lower()
        assert f"{N_SAE:,}" in msg


class TestProvenanceClausesAreVocabularyStatements:
    """Both provenance clauses describe a VOCABULARY; without one they must not print.

    The regression this pins: an if/else refactor of the no-vocab early return once
    let ``vocab_source`` / ``consolidated_away`` leak onto the uncompacted branch —
    silently changing SFT's startup line for every uncompacted-base run (SFT passes
    ``vocab_source`` unconditionally), an E3 no-cluster byte-identity violation.
    """

    def test_uncompacted_discards_vocab_source_exactly_like_the_SFT_call(self, full_width_model):
        # The verbatim SFT call shape (conditional_encdec_sft.py) against a
        # full-width model: the emitted line must be the bare no-compaction
        # sentence, with NO "Vocabulary ..." tail.
        msg = format_compaction_summary(
            full_width_model,
            vocab_source="inherited from the base checkpoint",
            no_vocab_reason="the base checkpoint is not compacted",
        )
        assert msg == (
            "Encoder feature dictionary: no compaction "
            f"({N_SAE:,} rows, the full SAE dictionary) — "
            "the base checkpoint is not compacted."
        )
        assert "Vocabulary" not in msg

    def test_uncompacted_discards_consolidated_away(self, full_width_model):
        msg = format_compaction_summary(
            full_width_model,
            no_vocab_reason=_REASON,
            consolidated_away=5,
        )
        assert "consolidated" not in msg

    def test_compacted_suppresses_a_zero_consolidated_away_count(self, model):
        with_zero = format_compaction_summary(model, no_vocab_reason=_REASON, consolidated_away=0)
        without = format_compaction_summary(model, no_vocab_reason=_REASON)
        assert with_zero == without
        assert "consolidated" not in with_zero

    def test_compacted_prints_a_positive_consolidated_away_count(self, model):
        msg = format_compaction_summary(model, no_vocab_reason=_REASON, consolidated_away=3)
        assert "Of the dropped rows, 3 were consolidated away by feature_clusters." in msg
